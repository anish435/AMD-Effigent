"""
Dynamic token budget estimator.

Estimates the optimal max_tokens for a given task based on its
complexity, type, and the hackathon task category. Setting tight
budgets on remote calls saves scored tokens without hurting accuracy.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from agent.models import RoutingDecision, Task

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category detection (mirrors executor._detect_category)
# ---------------------------------------------------------------------------

def detect_category(prompt: str) -> str:
    """Detect task category for budget allocation."""
    p = prompt.lower()

    # Code (debug + generation) — needs most tokens for complete code
    if any(kw in p for kw in ["write a function", "write a program", "implement a",
                               "write code", "write a python", "write a class",
                               "write a script", "generate code", "create a function",
                               "write a method", "code to",
                               "debug", "fix the bug", "bug in", "fix this code",
                               "what is wrong", "find the error", "buggy"]):
        return "code"

    # Sentiment — short label + brief justification
    if any(kw in p for kw in ["sentiment", "positive or negative", "classify the",
                               "is this positive", "is this negative", "tone of"]):
        return "sentiment"

    # NER — structured extraction
    if any(kw in p for kw in ["named entit", "extract entit", " ner ",
                               "identify the entit", "extract the names",
                               "person, org", "entities in", "entities", "entity", "entit",
                               "locations and", "organizations and"]):
        return "ner"

    # Summarisation — typically 1-3 sentences
    if any(kw in p for kw in ["summarise", "summarize", "summary of", "condense",
                               "in one sentence", "in a few words", "tldr"]):
        return "summarization"

    # Math — step-by-step + answer
    if any(kw in p for kw in ["calculate", "compute", "what is the value",
                               "original price", "compound interest",
                               "how much", "how many", "percentage", "profit",
                               "total distance", "total cost", "interest rate",
                               "probability of", "discount"]):
        return "math"

    # Simple arithmetic (just a number)
    if re.search(r"\d+\s*[\+\-\*\/\^]\s*\d+", p):
        return "simple_math"

    # Logic — step-by-step reasoning
    if any(kw in p for kw in ["logic", "deduc", "if all", "must be true",
                               "can we conclude", "constraint",
                               "puzzle", "who has", "given that"]):
        return "logic"

    # Explanation
    if any(kw in p for kw in ["explain", "describe", "how does", "what is",
                               "what causes", "what are", "concept of"]):
        return "explanation"

    return "general"


# ---------------------------------------------------------------------------
# Config-driven Budget allocation
# ---------------------------------------------------------------------------

_DEFAULT_CATEGORY_BUDGETS = {
    "code":          400,
    "sentiment":     80,
    "ner":           150,
    "summarization": 100,
    "math":          200,
    "simple_math":   30,
    "logic":         200,
    "explanation":   200,
    "general":       150,
    "default":       150,
}

def _load_category_budgets() -> dict[str, int]:
    """Loads budgets dynamically from BUDGET_CONFIG_PATH or local budgets.json."""
    config_path_str = os.getenv("BUDGET_CONFIG_PATH", "budgets.json")
    config_path = Path(config_path_str)
    
    if not config_path.exists():
        # Fallback search paths
        paths_to_check = [
            Path("budgets.json"),
            Path("d:/Projects/AMD_Effgent/hackathon-agent/budgets.json"),
            Path("/app/budgets.json")
        ]
        for p in paths_to_check:
            if p.exists():
                config_path = p
                break
                
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {k: int(v) for k, v in data.items()}
        except Exception as e:
            logger.warning("Failed to load budgets config from %s: %s. Using default budgets.", config_path, e)
            
    return _DEFAULT_CATEGORY_BUDGETS


def resolve_budget(
    task: Task,
    category: str,
    complexity_score: float,
    override: Optional[int] = None
) -> int:
    """
    Resolve token budget for a task dynamically, loading configuration and applying overrides.
    
    Tuned to satisfy accuracy requirements while keeping token usage capped.
    """
    # 1. Check override in parameters or task metadata
    budget_val = None
    if override is not None:
        budget_val = override
    elif task.metadata and ("max_tokens" in task.metadata or "max_new_tokens" in task.metadata):
        budget_val = task.metadata.get("max_tokens") or task.metadata.get("max_new_tokens")
        
    if budget_val is not None:
        try:
            return int(budget_val)
        except (ValueError, TypeError):
            pass

    # 2. Load category budgets
    budgets = _load_category_budgets()
    base_budget = budgets.get(category, budgets.get("default", 150))

    # Adjust for long prompts (they usually need longer answers)
    word_count = len(task.prompt.split())
    if word_count > 50 and category not in ("simple_math", "sentiment"):
        base_budget = max(base_budget, 250)

    # 3. Adjust for complexity (moderate bump)
    adjusted_budget = int(base_budget * (1.0 + 0.3 * complexity_score))

    # 4. Cap at MAX_TOKENS_CEILING
    ceiling_str = os.getenv("MAX_TOKENS_CEILING", "500")
    try:
        ceiling = int(ceiling_str)
    except ValueError:
        ceiling = 500

    return min(adjusted_budget, ceiling)


def estimate_token_budget(task: Task, decision: RoutingDecision) -> int:
    """Backward-compatible wrapper for estimate_token_budget calls."""
    category = detect_category(task.prompt)
    return resolve_budget(task, category, decision.complexity_score)
