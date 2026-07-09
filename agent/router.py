"""
Routing brain for the Hybrid Token-Efficient Routing Agent.

Combines a zero-cost heuristic classifier with a local Ollama tiebreaker
for ambiguous cases, deciding whether each task should be handled by the
local model (free) or the remote Fireworks AI model (costly).
"""

from __future__ import annotations

import logging
import os
import re
import requests
from typing import Any, Optional

from agent.config import RouterConfig
from agent.models import (
    Route,
    RoutingDecision,
    Task,
    TaskDifficulty,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal keywords — curated lists for heuristic classification
# ---------------------------------------------------------------------------

_COMPLEX_KEYWORDS: set[str] = {
    # Reasoning & logic
    "reason", "reasoning", "explain", "analyze", "analyse", "evaluate",
    "compare", "contrast", "critique", "prove", "proof", "derive",
    "deduce", "infer", "justify", "argue", "hypothesis",
    # Math
    "calculate", "compute", "solve", "equation", "integral", "derivative",
    "probability", "statistics", "algebra", "geometry", "theorem",
    "mathematical", "formula", "irrational",
    # Code & engineering
    "implement", "debug", "refactor", "algorithm", "function",
    "class", "optimize", "complexity", "backpropagation", "neural",
    "api", "endpoint", "authentication", "schema", "architect",
    "design", "system",
    # Multi-step markers
    "step", "first", "then", "finally", "multi-step",
    "strategy", "plan", "outline",
    # Creative / long-form
    "essay", "story", "compose", "draft", "comprehensive",
    "detailed", "in-depth", "thorough", "elaborate",
    # Generalization signals
    "generalize", "generalise", "extend", "apply",
}

_SIMPLE_KEYWORDS: set[str] = {
    "translate", "summarize", "summarise", "define", "what is",
    "who is", "when was", "where is", "yes or no", "true or false",
    "short answer", "brief", "one word",
    "hello", "hi", "thanks", "thank you", "greet",
    "capital", "name",
    # Additional patterns for local routing
    "how many", "how old", "how far", "how long",
    "what color", "what colour", "what year", "what day",
    "meaning of", "definition of", "synonym", "antonym",
    "spell", "abbreviation", "acronym",
    "convert", "temperature", "currency",
    "largest", "smallest", "tallest", "fastest",
    "president", "founder", "inventor", "author",
    "continent", "country", "city", "planet",
    "simple", "basic", "quick", "easy",
}

_STRUCTURED_OUTPUT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bjson\b", re.IGNORECASE),
    re.compile(r"\btable\b", re.IGNORECASE),
    re.compile(r"\bcsv\b", re.IGNORECASE),
    re.compile(r"\bmarkdown\b", re.IGNORECASE),
    re.compile(r"\byaml\b", re.IGNORECASE),
    re.compile(r"\bxml\b", re.IGNORECASE),
]

_MULTI_PART_PATTERN = re.compile(
    r"(\d+[\.\)]\s)|(\b(and also|additionally|furthermore|moreover)\b)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Ollama Availability Check
# ---------------------------------------------------------------------------

_OLLAMA_ALIVE: Optional[bool] = None

def _is_ollama_available() -> bool:
    """Checks if local Ollama service is reachable, launches it if not running, and caches result."""
    global _OLLAMA_ALIVE
    if _OLLAMA_ALIVE is not None:
        return _OLLAMA_ALIVE

    import subprocess
    import time

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = f"{ollama_host}/api/tags"

    # 1. Quick initial ping
    try:
        res = requests.get(url, timeout=1)
        if res.status_code == 200:
            _OLLAMA_ALIVE = True
            return _OLLAMA_ALIVE
    except Exception:
        pass

    # 2. Spawn serve in the background if it's not running
    logger = logging.getLogger(__name__)
    logger.info("Local Ollama service not running. Launching in-container 'ollama serve'...")

    start_time = time.monotonic()
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True
        )

        # Wait up to 30s for the port to become healthy
        for i in range(30):
            try:
                res = requests.get(url, timeout=1)
                if res.status_code == 200:
                    elapsed = time.monotonic() - start_time
                    logger.info("Ollama server started successfully and is ready in %.1fs.", elapsed)
                    _OLLAMA_ALIVE = True
                    return _OLLAMA_ALIVE
            except Exception:
                time.sleep(1)

        elapsed = time.monotonic() - start_time
        logger.warning("Ollama serve launched but failed to become healthy within %.1fs.", elapsed)
    except Exception as e:
        logger.error("Failed to launch in-container Ollama binary: %s", e)

    _OLLAMA_ALIVE = False
    return _OLLAMA_ALIVE


def _ollama_classify(prompt: str) -> Optional[str]:
    """Calls Ollama to classify the prompt difficulty into easy, medium, or hard."""
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = f"{ollama_host}/api/generate"
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    
    system_instruction = (
        "You are a task difficulty classifier. Classify the user prompt as either 'easy', 'medium', or 'hard'.\n"
        "Guidelines:\n"
        "- easy: simple lookup, math, short factual answer\n"
        "- medium: needs reasoning, multi-step, moderate context\n"
        "- hard: complex reasoning, ambiguous, long context, high accuracy needed\n"
        "Respond ONLY with one of these three lowercase words: easy, medium, hard. "
        "Do not include any punctuation, conversational filler, or formatting. Your reply must be exactly one word."
    )
    
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_instruction,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
        result = data.get("response", "").strip().lower()
        
        result_clean = "".join(c for c in result if c.isalnum() or c.isspace()).strip()
        words = result_clean.split()
        first_word = words[0] if words else ""
        
        if first_word in ["easy", "medium", "hard"]:
            return first_word
    except Exception as e:
        logger.warning("Ollama connection lost during routing: %s. Disabling Ollama routing.", e)
        global _OLLAMA_ALIVE
        _OLLAMA_ALIVE = False
        
    return None


# ---------------------------------------------------------------------------
# Heuristic + Local LLM Router
# ---------------------------------------------------------------------------

class HeuristicRouter:
    """
    Hybrid router that buckets obvious cases instantly via rules,
    and falls back to local Ollama for ambiguous tiebreakers.
    """

    _WEIGHTS: dict[str, float] = {
        "length": 0.10,
        "complex_keywords": 0.30,
        "simple_keywords": 0.15,
        "structured_output": 0.10,
        "multi_part": 0.10,
        "question_depth": 0.05,
        "sentence_count": 0.20,
    }

    def __init__(self, config: RouterConfig) -> None:
        self._config = config

    def route(self, task: Task) -> RoutingDecision:
        """Classify a task and decide where to route it."""
        from agent.budget import detect_category
        category = detect_category(task.prompt)
        
        signals = self._compute_signals(task.prompt)
        score = self._aggregate(signals)
        
        # Check if Ollama is available
        ollama_active = _is_ollama_available()
        
        route_decision = Route.REMOTE
        difficulty = TaskDifficulty.MODERATE
        reasoning = ""
        
        if not ollama_active:
            # 100% remote fallback path
            route_decision = Route.REMOTE
            difficulty = self._score_to_difficulty(score)
            reasoning = f"Ollama unreachable. Falling back to 100% remote routing. Heuristic score: {score:.3f}"
        else:
            # Check for categories that struggle under local 1.5B model
            if category in ["math", "logic", "ner"]:
                route_decision = Route.REMOTE
                difficulty = TaskDifficulty.SIMPLE
                reasoning = f"Forcing category '{category}' to remote simple model to ensure direct answers. Heuristic score: {score:.3f}"
            # Heuristic bucketing
            elif score < 0.35:
                route_decision = Route.LOCAL
                difficulty = TaskDifficulty.SIMPLE
                reasoning = f"Heuristic score {score:.3f} is obviously easy (SIMPLE)."
            elif score >= 0.65:
                route_decision = Route.REMOTE
                difficulty = TaskDifficulty.COMPLEX
                reasoning = f"Heuristic score {score:.3f} is obviously complex (COMPLEX)."
            else:
                # Ambiguous middle band -> call Ollama tiebreaker
                ollama_verdict = _ollama_classify(task.prompt)
                if ollama_verdict == "easy":
                    route_decision = Route.LOCAL
                    difficulty = TaskDifficulty.SIMPLE
                    reasoning = f"Ambiguous heuristic score {score:.3f}. Ollama tiebreaker classified as easy."
                elif ollama_verdict == "medium":
                    route_decision = Route.REMOTE
                    difficulty = TaskDifficulty.MODERATE
                    reasoning = f"Ambiguous heuristic score {score:.3f}. Ollama tiebreaker classified as medium."
                elif ollama_verdict == "hard":
                    route_decision = Route.REMOTE
                    difficulty = TaskDifficulty.COMPLEX
                    reasoning = f"Ambiguous heuristic score {score:.3f}. Ollama tiebreaker classified as hard."
                else:
                    # Ollama call failed or returned invalid response -> fallback to heuristic threshold
                    route_decision = (
                        Route.REMOTE
                        if score >= self._config.complexity_threshold
                        else Route.LOCAL
                    )
                    difficulty = self._score_to_difficulty(score)
                    reasoning = f"Ambiguous heuristic score {score:.3f}. Ollama tiebreaker failed. Used heuristic threshold."

        # Add category to signals dict so it is captured in AgentResponse
        signals["category"] = category

        logger.debug(
            "Router decision for task %s: %s (score=%.3f, category=%s)",
            task.id, route_decision.value, score, category
        )
        
        return RoutingDecision(
            route=route_decision,
            category=category,
            complexity_score=round(score, 4),
            difficulty=difficulty,
            reason=reasoning,
            signals=signals,
        )

    # ----- signal computation -----

    def _compute_signals(self, prompt: str) -> dict[str, Any]:
        """Compute individual heuristic signals from the prompt."""
        prompt_lower = prompt.lower()
        word_count = len(prompt.split())

        return {
            "length": self._length_signal(word_count),
            "complex_keywords": self._keyword_signal(prompt_lower, _COMPLEX_KEYWORDS),
            "simple_keywords": self._keyword_signal(prompt_lower, _SIMPLE_KEYWORDS),
            "structured_output": self._structured_output_signal(prompt),
            "multi_part": self._multi_part_signal(prompt),
            "question_depth": self._question_depth_signal(prompt),
            "sentence_count": self._sentence_count_signal(prompt),
            "word_count": word_count,  # informational, not scored
        }

    @staticmethod
    def _length_signal(word_count: int) -> float:
        """Longer prompts tend to be more complex. Sigmoid-ish curve."""
        if word_count <= 10:
            return 0.1
        if word_count <= 30:
            return 0.3
        if word_count <= 80:
            return 0.5
        if word_count <= 150:
            return 0.7
        return 0.9

    @staticmethod
    def _keyword_signal(prompt_lower: str, keywords: set[str]) -> float:
        """Fraction of keyword set that appears in the prompt (capped at 1)."""
        # Match using word boundaries to avoid substring matching bugs
        hits = 0
        for kw in keywords:
            # Escape to prevent regex issues
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, prompt_lower):
                hits += 1
        # Normalize: 2+ hits is a strong signal
        return min(hits / 2.0, 1.0)

    @staticmethod
    def _structured_output_signal(prompt: str) -> float:
        """Does the prompt request structured output?"""
        hits = sum(1 for pat in _STRUCTURED_OUTPUT_PATTERNS if pat.search(prompt))
        return min(hits / 2.0, 1.0)

    @staticmethod
    def _multi_part_signal(prompt: str) -> float:
        """Does the prompt contain multiple sub-questions or steps?"""
        matches = _MULTI_PART_PATTERN.findall(prompt)
        return min(len(matches) / 3.0, 1.0)

    @staticmethod
    def _question_depth_signal(prompt: str) -> float:
        """Count question marks as a rough proxy for complexity."""
        q_count = prompt.count("?")
        if q_count <= 1:
            return 0.1
        if q_count <= 3:
            return 0.5
        return 0.9

    @staticmethod
    def _sentence_count_signal(prompt: str) -> float:
        """More sentences usually means a more complex, multi-part request."""
        sentences = re.split(r'[.!?]+', prompt)
        count = len([s for s in sentences if s.strip()])
        if count <= 1:
            return 0.05
        if count <= 2:
            return 0.2
        if count <= 4:
            return 0.5
        if count <= 6:
            return 0.75
        return 1.0

    # ----- aggregation -----

    def _aggregate(self, signals: dict[str, Any]) -> float:
        """Weighted combination of signals into a single complexity score."""
        score = 0.0
        for name, weight in self._WEIGHTS.items():
            value = signals.get(name, 0.0)
            if name == "simple_keywords":
                score += weight * (1.0 - value)
            else:
                score += weight * value
        return max(0.0, min(1.0, score))

    @staticmethod
    def _score_to_difficulty(score: float) -> TaskDifficulty:
        if score < 0.35:
            return TaskDifficulty.SIMPLE
        if score < 0.65:
            return TaskDifficulty.MODERATE
        return TaskDifficulty.COMPLEX
