"""
Configuration management for the Hybrid Token-Efficient Routing Agent.

All settings are loaded from environment variables (with .env file support)
so the container can be configured at runtime without code changes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root (if it exists)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class FireworksConfig:
    """Settings for the remote Fireworks AI API."""
    api_key: str = field(
        default_factory=lambda: os.getenv("FIREWORKS_API_KEY", "")
    )
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
        )
    )
    model: str = field(
        default_factory=lambda: os.getenv(
            "FIREWORKS_MODEL",
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
        )
    )
    allowed_models: list[str] = field(
        default_factory=lambda: [
            m.strip()
            for m in os.getenv("ALLOWED_MODELS", "").split(",")
            if m.strip()
        ]
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("FIREWORKS_TEMPERATURE", "0.1"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("FIREWORKS_MAX_TOKENS", "512"))
    )


@dataclass(frozen=True)
class LocalModelConfig:
    """Settings for the local model."""
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "LOCAL_MODEL_NAME", "google/gemma-2-2b-it"
        )
    )
    device: str = field(
        default_factory=lambda: os.getenv("LOCAL_MODEL_DEVICE", "auto")
    )
    max_new_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("LOCAL_MODEL_MAX_NEW_TOKENS", "512")
        )
    )
    temperature: float = field(
        default_factory=lambda: float(
            os.getenv("LOCAL_MODEL_TEMPERATURE", "0.1")
        )
    )
    torch_dtype: str = field(
        default_factory=lambda: os.getenv("LOCAL_MODEL_DTYPE", "auto")
    )


@dataclass(frozen=True)
class RouterConfig:
    """Thresholds that control routing decisions."""
    # Complexity score above this sends the task to the remote model
    complexity_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("ROUTER_COMPLEXITY_THRESHOLD", "0.6")
        )
    )
    # If local model confidence is below this, fallback to remote
    confidence_fallback_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("ROUTER_CONFIDENCE_FALLBACK_THRESHOLD", "0.2")
        )
    )


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration — aggregates all sub-configs."""
    fireworks: FireworksConfig = field(default_factory=FireworksConfig)
    local_model: LocalModelConfig = field(default_factory=LocalModelConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    cache_enabled: bool = field(
        default_factory=lambda: os.getenv("CACHE_ENABLED", "true").lower()
        in ("true", "1", "yes")
    )
    compression_enabled: bool = field(
        default_factory=lambda: os.getenv("COMPRESSION_ENABLED", "true").lower()
        in ("true", "1", "yes")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    def validate(self) -> list[str]:
        """Return a list of configuration warnings/errors."""
        issues: list[str] = []
        if not self.fireworks.api_key:
            issues.append(
                "FIREWORKS_API_KEY is not set — remote model calls will fail."
            )
        if not (0.0 <= self.router.complexity_threshold <= 1.0):
            issues.append(
                "ROUTER_COMPLEXITY_THRESHOLD must be between 0.0 and 1.0."
            )
        if not (0.0 <= self.router.confidence_fallback_threshold <= 1.0):
            issues.append(
                "ROUTER_CONFIDENCE_FALLBACK_THRESHOLD must be between 0.0 and 1.0."
            )
        return issues


def get_sorted_allowed_models(config: AppConfig) -> list[str]:
    """Sort ALLOWED_MODELS from smallest to largest for token efficiency."""
    if not config.fireworks.allowed_models:
        return []

    import re

    def model_priority(model_id: str) -> int:
        model_lower = model_id.lower()
        
        # 1. Look for numeric parameter size (e.g., 8b, 20b, 70b, 120b)
        match = re.search(r"(\d+)b", model_lower)
        if match:
            return int(match.group(1))
            
        # 2. Non-numeric custom sizing keywords
        custom_priorities = {
            "scout": 3,
            "mini": 5,
            "instant": 5,
            "small": 10,
            "flash": 15,
            "pro": 30,
            "maverick": 40,
            "large": 70,
        }
        
        for keyword, prio in custom_priorities.items():
            if keyword in model_lower:
                return prio
                
        return 999  # Unknown models last

    return sorted(config.fireworks.allowed_models, key=model_priority)


def get_model_for_difficulty(config: AppConfig, difficulty: str) -> str:
    """Select the best model from ALLOWED_MODELS based on task difficulty."""
    logger = logging.getLogger(__name__)
    allowed = get_sorted_allowed_models(config)
    
    if not allowed:
        logger.debug("ALLOWED_MODELS not set — using default Fireworks model: %s", config.fireworks.model)
        return config.fireworks.model

    # difficulty is simple/moderate/complex (mapped to easy/medium/hard)
    diff_str = str(difficulty).lower()
    
    if diff_str in ("simple", "easy"):
        selected = allowed[0]
    elif diff_str in ("moderate", "medium"):
        mid_idx = len(allowed) // 2
        selected = allowed[mid_idx]
    else:  # complex / hard
        selected = allowed[-1]

    logger.debug("Selected model '%s' for difficulty '%s'", selected, difficulty)
    return selected


def load_config() -> AppConfig:
    """Create and validate the application config from the environment."""
    return AppConfig()


def get_resolved_model(config: AppConfig) -> str:
    """Get default/fallback resolved model ID."""
    allowed = get_sorted_allowed_models(config)
    if allowed:
        return allowed[0]
    return config.fireworks.model
