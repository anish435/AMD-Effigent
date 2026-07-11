"""
Executors for local and remote model inference.

- RuleBasedExecutor: Handles simple tasks with pattern matching (zero cost, no GPU).
- LocalExecutor:  Runs a local Ollama model for inference (zero scored tokens).
- RemoteExecutor: Calls the Fireworks AI API via OpenAI-compatible client.
- HybridExecutor: Orchestrates all three with:
    • Response caching (identical prompts = free)
    • Rule-based fast path (simple tasks = instant + free)
    • Prompt compression (fewer remote tokens)
    • Dynamic token budgeting (tight max_tokens per task)
    • Cascading fallback (rules → local model → verify → remote)
"""

from __future__ import annotations

import logging
import os
import re
import time
import requests
from typing import Optional

from openai import OpenAI

from agent.budget import resolve_budget
from agent.cache import ResponseCache
from agent.compressor import compress_prompt
from agent.config import AppConfig, get_model_for_difficulty, get_sorted_allowed_models
from agent.models import (
    ExecutionResult,
    Route,
    RoutingDecision,
    Task,
    TokenUsage,
)
from agent.tracker import TokenTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category system prompts (hackathon-specific)
# ---------------------------------------------------------------------------

_CATEGORY_PROMPTS: dict[str, str] = {
    "sentiment": "Classify the sentiment (positive/negative/neutral). State the label first, then briefly justify.",
    "ner": "Extract the requested named entities. Output ONLY the extracted entity text separated by spaces, in the order they appear, without types, without labels, and without any other text.",
    "summarization": "Summarize the text concisely. Follow any length or format constraints given.",
    "code": "Answer the coding task directly. For debugging, show the fix. For code generation, write correct code.",
    "math": "Solve the math problem. Output ONLY the final numerical answer, no explanation, no steps, and no work shown.",
    "logic": "Reason through the constraints. Output ONLY the final direct answer, no explanation, and no reasoning steps shown.",
    "explanation": "Answer in 1-2 sentences max, only the core fact, no elaboration unless explicitly asked to explain.",
    "general": "Answer in 1-2 sentences max, only the core fact, no elaboration unless explicitly asked to explain.",
}


# ---------------------------------------------------------------------------
# Remote executor (Fireworks AI)
# ---------------------------------------------------------------------------

class RemoteExecutor:
    """Calls the remote API through the OpenAI-compatible SDK with timeouts/retries."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config.fireworks
        self._app_config = config
        self._client = OpenAI(
            base_url=self._config.base_url,
            api_key=self._config.api_key,
        )

    def execute(
        self,
        task: Task,
        max_tokens_override: Optional[int] = None,
        compress: bool = True,
        difficulty: str = "moderate",
        category: Optional[str] = None,
    ) -> ExecutionResult:
        """Send the task to the remote model with retry and timeout logic."""
        prompt = task.prompt

        # Compress prompt to save input tokens
        if compress:
            prompt = compress_prompt(prompt)

        max_tokens = max_tokens_override or self._config.max_tokens

        # Category-aware system prompt for better accuracy
        from agent.budget import detect_category
        if category is None:
            category = detect_category(task.prompt)
        system_prompt = _CATEGORY_PROMPTS.get(category, "Answer accurately and concisely.")
        # Enforce caveman style directly in the system prompt to keep completions token-efficient
        system_prompt = (
            system_prompt +
            " Answer directly. No preamble, no conversational filler, no restating the question. Just the direct answer."
        )

        model = get_model_for_difficulty(self._app_config, difficulty, category)

        # Safety net: largest model (usually reasoning gpt-oss-120b) needs plenty of tokens
        # to generate reasoning. If we restrict it, it will truncate early and fail accuracy.
        allowed = get_sorted_allowed_models(self._app_config)
        if allowed and model == allowed[-1]:
            max_tokens = max(max_tokens, 1024)

        logger.info(
            "Remote execution for task %s via %s (max_tokens=%d, category=%s)",
            task.id, model, max_tokens, category,
        )
        
        start = time.perf_counter()
        
        retries = 2
        backoff = 1.0  # 1s initial backoff
        last_exception = None

        for attempt in range(retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self._config.temperature,
                    max_tokens=max_tokens,
                    timeout=10.0,  # 10s maximum timeout per call
                )

                elapsed_ms = (time.perf_counter() - start) * 1000
                usage = response.usage
                output_text = response.choices[0].message.content or ""

                return ExecutionResult(
                    output=output_text.strip(),
                    route_used=Route.REMOTE,
                    token_usage=TokenUsage(
                        prompt_tokens=usage.prompt_tokens if usage else 0,
                        completion_tokens=usage.completion_tokens if usage else 0,
                        total_tokens=usage.total_tokens if usage else 0,
                    ),
                    confidence=1.0,
                    latency_ms=elapsed_ms,
                    fallback_triggered=False,
                )

            except Exception as exc:
                last_exception = exc
                logger.warning(
                    "Remote execution attempt %d failed for task %s: %s",
                    attempt + 1, task.id, exc
                )
                if attempt < retries:
                    time.sleep(backoff)
                    backoff *= 2.0  # Exponential backoff (1s, 2s)
                else:
                    break

        # Fallback if all attempts fail
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error("All remote execution attempts failed for task %s: %s", task.id, last_exception)
        return ExecutionResult(
            output=f"[ERROR] Remote model call failed: {last_exception}",
            route_used=Route.REMOTE,
            token_usage=TokenUsage(),
            confidence=0.0,
            latency_ms=elapsed_ms,
            fallback_triggered=False,
        )


# ---------------------------------------------------------------------------
# Program-aided math verification helpers
# ---------------------------------------------------------------------------

def safe_eval(expr: str) -> Optional[float]:
    """Safely evaluates a basic arithmetic expression."""
    # Sanitize: allow only numbers, basic arithmetic operators, brackets, and spaces
    if not re.match(r"^[0-9\.\+\-\*\/\(\)\s]+$", expr):
        return None
    try:
        val = eval(expr, {"__builtins__": None}, {})
        return float(val)
    except Exception:
        return None

def extract_final_number(text: str) -> Optional[float]:
    """Extracts the final numeric value stated in the model output."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None

def verify_math_answer(prompt: str, answer: str) -> Optional[bool]:
    """
    Attempts to programmatically verify an arithmetic task's answer.
    Returns:
      True if verification succeeds (answers match).
      False if verification fails (answers mismatch).
      None if the expression or answer cannot be parsed/verified.
    """
    try:
        # Extract expression matching numbers and operators
        expr_match = re.search(r"\d+(?:\.\d+)?(?:\s*[\+\-\*\/\(\)]+\s*\d+(?:\.\d+)?)+", prompt)
        if not expr_match:
            return None
        
        expr = expr_match.group(0)
        computed = safe_eval(expr)
        if computed is None:
            return None
            
        model_num = extract_final_number(answer)
        if model_num is None:
            return None
            
        match = abs(computed - model_num) < 1e-4
        if not match:
            logger.info("Math verification mismatch: computed %.4f vs model %.4f", computed, model_num)
        else:
            logger.info("Math verification match: computed %.4f vs model %.4f", computed, model_num)
        return match
    except Exception as exc:
        logger.warning("Error in math verification: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Local executor (Ollama connector)
# ---------------------------------------------------------------------------

class LocalExecutor:
    """
    Runs a local Ollama model for inference (zero scored tokens).
    
    Raises exceptions on unreachable connection so that HybridExecutor can fallback remote.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config.local_model
        self._model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

    def execute(
        self,
        task: Task,
        max_new_tokens_override: Optional[int] = None,
    ) -> ExecutionResult:
        """Run the task through the local Ollama model."""
        start = time.perf_counter()
        
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        url = f"{ollama_host}/api/generate"
        
        max_new_tokens = max_new_tokens_override or self._config.max_new_tokens
        
        payload = {
            "model": self._model_name,
            "prompt": task.prompt,
            "stream": False,
            "options": {
                "temperature": self._config.temperature,
                "num_predict": max_new_tokens
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            output_text = data.get("response", "").strip()
            
            # Approximate token counting (1 word ≈ 1.33 tokens)
            prompt_words = len(task.prompt.split())
            completion_words = len(output_text.split())
            prompt_tokens = int(prompt_words / 0.75)
            completion_tokens = int(completion_words / 0.75)
            total_tokens = prompt_tokens + completion_tokens
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            return ExecutionResult(
                output=output_text,
                route_used=Route.LOCAL,
                token_usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
                confidence=1.0,  # Default to high confidence; verified by OutputVerifier
                latency_ms=elapsed_ms,
                fallback_triggered=False,
            )
            
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("Local Ollama execution failed: %s", exc)
            # Raise exception to trigger the coordinator remote fallback
            raise RuntimeError(f"Local Ollama unreachable or failed: {exc}") from exc

    def execute_with_consensus(
        self,
        task: Task,
        num_samples: int = 3,
        max_new_tokens_override: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Run the task through the local Ollama model multiple times with elevated temperature
        to measure consensus agreement across samples.
        """
        # If dynamic governor returns 1, fall back directly to original single-shot execution
        if num_samples <= 1:
            logger.info("Adaptive Governor set sample count <= 1. Running single-shot local execution.")
            return self.execute(task, max_new_tokens_override)
            
        start = time.perf_counter()
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        url = f"{ollama_host}/api/generate"
        max_new_tokens = max_new_tokens_override or self._config.max_new_tokens
        
        # Elevated temperature for diversity in sampling
        payload = {
            "model": self._model_name,
            "prompt": task.prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": max_new_tokens
            }
        }
        
        samples: list[str] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for idx in range(num_samples):
            try:
                response = requests.post(url, json=payload, timeout=15)
                response.raise_for_status()
                data = response.json()
                output_text = data.get("response", "").strip()
                samples.append(output_text)
                
                # Approximate token counting
                prompt_words = len(task.prompt.split())
                completion_words = len(output_text.split())
                total_prompt_tokens += int(prompt_words / 0.75)
                total_completion_tokens += int(completion_words / 0.75)
            except Exception as exc:
                logger.warning("Local consensus sample %d of %d failed: %s", idx + 1, num_samples, exc)
                if len(samples) == 0:
                    raise exc
                    
        if not samples:
            raise RuntimeError("All local consensus samples failed")
            
        def normalize(t: str) -> str:
            # strip non-alphanumeric and lowercase
            return re.sub(r"[^a-z0-9]", "", t.lower())
            
        def is_similar(a: str, b: str) -> bool:
            a_norm, b_norm = normalize(a), normalize(b)
            if not a_norm or not b_norm:
                return a_norm == b_norm
            # If short (e.g. less than 15 chars), require exact normalized match
            if len(a_norm) < 15 or len(b_norm) < 15:
                return a_norm == b_norm
            # Otherwise use Jaccard word-overlap similarity (0.60 threshold)
            a_words = set(re.findall(r"\w+", a.lower()))
            b_words = set(re.findall(r"\w+", b.lower()))
            union = a_words.union(b_words)
            if not union:
                return False
            overlap = len(a_words.intersection(b_words)) / len(union)
            return overlap >= 0.60
            
        # Group similar samples into clusters
        clusters: list[list[str]] = []
        for sample in samples:
            added = False
            for cluster in clusters:
                if is_similar(sample, cluster[0]):
                    cluster.append(sample)
                    added = True
                    break
            if not added:
                clusters.append([sample])
                
        # Find majority/plurality cluster
        clusters.sort(key=len, reverse=True)
        majority_cluster = clusters[0]
        majority_answer = majority_cluster[0]
        
        consensus_confidence = len(majority_cluster) / num_samples
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        logger.info(
            "Task %s consensus: samples=%d majority_size=%d confidence=%.2f elapsed=%.0fms",
            task.id, num_samples, len(majority_cluster), consensus_confidence, elapsed_ms
        )
        
        return ExecutionResult(
            output=majority_answer,
            route_used=Route.LOCAL,
            token_usage=TokenUsage(
                prompt_tokens=int(total_prompt_tokens / num_samples),
                completion_tokens=int(total_completion_tokens / num_samples),
                total_tokens=int((total_prompt_tokens + total_completion_tokens) / num_samples),
            ),
            confidence=consensus_confidence,
            latency_ms=elapsed_ms,
            fallback_triggered=False,
        )


# ---------------------------------------------------------------------------
# Rule-based executor (dynamic computation only)
# ---------------------------------------------------------------------------

class RuleBasedExecutor:
    """
    Handles tasks that can be solved by pure computation — no model needed.
    """

    _MATH_PATTERN = re.compile(
        r"(?:what is|calculate|compute|solve|evaluate)?\s*(\d+(?:\.\d+)?)\s*"
        r"([\+\-\*\/\^])\s*(\d+(?:\.\d+)?)",
        re.I,
    )

    def try_execute(self, task: Task) -> Optional[ExecutionResult]:
        """Try to answer the task with pure computation."""
        prompt = task.prompt.strip()

        word_count = len(prompt.split())
        if word_count <= 20:
            result = self._try_math(prompt)
            if result is not None:
                return result

        return None

    def _make_result(self, output: str) -> ExecutionResult:
        """Create a zero-cost execution result."""
        return ExecutionResult(
            output=output,
            route_used=Route.LOCAL,
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            confidence=1.0,
            latency_ms=0.0,
            fallback_triggered=False,
        )

    def _try_math(self, prompt: str) -> Optional[ExecutionResult]:
        match = self._MATH_PATTERN.search(prompt)
        if not match:
            return None

        try:
            a = float(match.group(1))
            op = match.group(2)
            b = float(match.group(3))

            if op == "+":
                result = a + b
            elif op == "-":
                result = a - b
            elif op == "*":
                result = a * b
            elif op == "/":
                result = a / b if b != 0 else float("inf")
            elif op == "^":
                result = a ** b
            else:
                return None
            
            answer = str(int(result)) if result == int(result) else str(result)
            return self._make_result(answer)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Output verifier (for cascading execution)
# ---------------------------------------------------------------------------

class OutputVerifier:
    """Lightweight output verification checks for local model responses."""

    @staticmethod
    def verify(task: Task, result: ExecutionResult) -> bool:
        """Returns True if the output passes basic quality checks."""
        output = result.output.strip()

        # 1. Empty or error output → fail
        if not output or output.startswith("[ERROR]"):
            return False

        # 2. Too short for the prompt complexity
        prompt_words = len(task.prompt.split())
        output_words = len(output.split())

        if prompt_words > 15 and output_words < 2:
            return False

        if prompt_words > 30 and output_words < max(3, prompt_words * 0.05):
            return False

        # 3. Repetition detection — if the output repeats itself excessively
        if OutputVerifier._has_excessive_repetition(output):
            return False

        # 4. Confidence-based check
        if result.confidence < 0.10:
            return False

        # 5. Gibberish detection — high ratio of non-alphanumeric chars
        alnum_ratio = sum(c.isalnum() or c.isspace() for c in output) / max(len(output), 1)
        if alnum_ratio < 0.5:
            return False

        return True

    @staticmethod
    def _has_excessive_repetition(text: str) -> bool:
        """Check if text contains excessive repetition (degenerate output)."""
        words = text.split()
        if len(words) < 10:
            return False

        from collections import Counter
        counts = Counter(words)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count / len(words) > 0.4:
            return True

        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        if trigrams:
            trigram_counts = Counter(trigrams)
            most_common_tri = trigram_counts.most_common(1)[0][1]
            if most_common_tri > 3 and most_common_tri / len(trigrams) > 0.3:
                return True

        return False


# ---------------------------------------------------------------------------
# Hybrid executor (orchestrator)
# ---------------------------------------------------------------------------

class HybridExecutor:
    """
    Orchestrates local and remote execution with full optimization stack:
    1. Cache check
    2. Rule check
    3. Route decision (local / remote)
    4. Execution and fallback
    """

    def __init__(self, config: AppConfig, tracker: Optional[TokenTracker] = None) -> None:
        self._config = config
        self._rules = RuleBasedExecutor()
        self._local = LocalExecutor(config)
        self._remote = RemoteExecutor(config)
        self._cache = ResponseCache(enabled=config.cache_enabled)
        self._verifier = OutputVerifier()
        self._fallback_threshold = config.router.consensus_confidence_threshold
        self._compression_enabled = config.compression_enabled
        self._tracker = tracker

    @property
    def cache(self) -> ResponseCache:
        return self._cache

    def execute(
        self, task: Task, decision: RoutingDecision
    ) -> ExecutionResult:
        """Execute the task with caching, rule checking, and hybrid routing."""
        category = decision.category
        difficulty = decision.difficulty.value
        
        # Determine baseline tokens for tracker
        prompt_words = len(task.prompt.split())
        
        # ── Step 1: Cache check ──
        cached = self._cache.get(task.prompt)
        if cached is not None:
            logger.info("Task %s: served from cache (0 tokens)", task.id)
            
            # Resolve default budget for baseline savings report
            token_budget = resolve_budget(task, category, decision.complexity_score)
            baseline_tokens = int(prompt_words / 0.75) + token_budget
            
            if self._tracker:
                self._tracker.log_free_resolution(
                    task_id=task.id,
                    resolution_type="cache hit",
                    reason="Found normalized prompt in response cache",
                    category=category,
                    difficulty=difficulty,
                    baseline_tokens=baseline_tokens
                )
            return cached

        # Compute dynamic token budget
        token_budget = resolve_budget(task, category, decision.complexity_score)
        baseline_tokens = int(prompt_words / 0.75) + token_budget

        # ── Step 2: Rule-based fast path ──
        rule_result = self._rules.try_execute(task)
        if rule_result is not None:
            logger.info("Task %s: answered by rule-based executor (0 tokens)", task.id)
            self._cache.put(task.prompt, rule_result)
            
            if self._tracker:
                self._tracker.log_free_resolution(
                    task_id=task.id,
                    resolution_type="rule-based",
                    reason="Solved via regex pattern matcher",
                    category=category,
                    difficulty=difficulty,
                    baseline_tokens=baseline_tokens
                )
            return rule_result

        # ── Step 3: Route to remote ──
        if decision.route == Route.REMOTE:
            result = self._remote.execute(
                task,
                max_tokens_override=token_budget,
                compress=self._compression_enabled,
                difficulty=difficulty,
                category=category,
            )
            from agent.formatter import strip_filler
            result.output = strip_filler(result.output)
            self._cache.put(task.prompt, result)
            
            if self._tracker:
                prompt_tokens = result.token_usage.prompt_tokens
                completion_tokens = result.token_usage.completion_tokens
                self._tracker.log_fireworks_call(
                    task_id=task.id,
                    model=get_model_for_difficulty(self._config, difficulty),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    category=category,
                    difficulty=difficulty,
                    baseline_tokens=baseline_tokens
                )
            return result

        # ── Step 4: Local model execution with cascading verification ──
        try:
            try:
                from agent.governor import get_current_sample_count
                num_samples = get_current_sample_count(self._config.router.total_runtime_budget)
            except Exception as gov_exc:
                logger.warning("Governor error: %s. Defaulting to 1 sample.", gov_exc)
                num_samples = 1

            try:
                local_result = self._local.execute_with_consensus(
                    task,
                    num_samples=num_samples,
                    max_new_tokens_override=token_budget,
                )
            except Exception as consensus_exc:
                logger.warning("Consensus sampling failed: %s. Falling back to single-shot execution.", consensus_exc)
                local_result = self._local.execute(
                    task,
                    max_new_tokens_override=token_budget,
                )
        except Exception as exc:
            logger.warning(
                "Task %s: local model execution failed entirely (%s) — routing to remote",
                task.id, exc,
            )
            result = self._remote.execute(
                task,
                max_tokens_override=token_budget,
                compress=self._compression_enabled,
                difficulty=difficulty,
                category=category,
            )
            from agent.formatter import strip_filler
            result.output = strip_filler(result.output)
            self._cache.put(task.prompt, result)
            
            if self._tracker:
                prompt_tokens = result.token_usage.prompt_tokens
                completion_tokens = result.token_usage.completion_tokens
                self._tracker.log_fireworks_call(
                    task_id=task.id,
                    model=get_model_for_difficulty(self._config, difficulty),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    category=category,
                    difficulty=difficulty,
                    baseline_tokens=baseline_tokens
                )
            return result

        # ── Step 5: Verify local output quality ──
        confidence_ok = local_result.confidence >= self._fallback_threshold
        verification_ok = self._verifier.verify(task, local_result)

        # Math programmatic verification check (Part B & C)
        math_verification_ok = True
        if category in ("math", "simple_math"):
            try:
                math_verified = verify_math_answer(task.prompt, local_result.output)
                if math_verified is False:
                    logger.info("Task %s: Program-aided math verification failed. Rejecting local answer.", task.id)
                    math_verification_ok = False
                elif math_verified is True:
                    logger.info("Task %s: Program-aided math verification succeeded. Boosting confidence.", task.id)
                    local_result.confidence = 1.0
                    confidence_ok = True
            except Exception as math_exc:
                logger.warning("Task %s: Program-aided math verification threw exception: %s. Treating as unverifiable.", task.id, math_exc)

        if confidence_ok and verification_ok and math_verification_ok:
            from agent.formatter import strip_filler
            local_result.output = strip_filler(local_result.output)
            self._cache.put(task.prompt, local_result)
            
            if self._tracker:
                self._tracker.log_free_resolution(
                    task_id=task.id,
                    resolution_type="local model",
                    reason=f"Answered by local model and verified successfully (confidence={local_result.confidence:.2f})",
                    category=category,
                    difficulty=difficulty,
                    baseline_tokens=baseline_tokens
                )
            return local_result

        # ── Step 6: Escalate to remote (fallback) ──
        reason = []
        if not confidence_ok:
            reason.append(f"confidence {local_result.confidence:.2f} < {self._fallback_threshold:.2f}")
        if not verification_ok:
            reason.append("output quality verification failed")
        if not math_verification_ok:
            reason.append("program-aided math verification failed")
        reason_str = " and ".join(reason)
        logger.info("Task %s: local answer rejected (%s) — escalating to remote", task.id, reason_str)

        result = self._remote.execute(
            task,
            max_tokens_override=token_budget,
            compress=self._compression_enabled,
            difficulty=difficulty,
            category=category,
        )
        result.fallback_triggered = True
        from agent.formatter import strip_filler
        result.output = strip_filler(result.output)
        self._cache.put(task.prompt, result)
        
        if self._tracker:
            prompt_tokens = result.token_usage.prompt_tokens
            completion_tokens = result.token_usage.completion_tokens
            self._tracker.log_fireworks_call(
                task_id=task.id,
                model=get_model_for_difficulty(self._config, difficulty),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                category=category,
                difficulty=difficulty,
                baseline_tokens=baseline_tokens
            )
        return result
