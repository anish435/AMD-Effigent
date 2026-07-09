"""
Token usage tracker and statistics reporter.

Collects per-task metrics (route chosen, tokens, latency) and produces
a rich summary report showing total scored tokens, routing distribution,
cache performance, and potential savings.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from agent.models import AgentResponse, Route

logger = logging.getLogger(__name__)

# ANSI color codes (safe on most terminals)
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"
_WHITE = "\033[97m"
_BG_GREEN = "\033[42m"
_BG_RED = "\033[41m"
_BG_YELLOW = "\033[43m"


def _supports_color() -> bool:
    """Check if the terminal supports ANSI colors."""
    if sys.platform == "win32":
        return True  # Modern Windows terminals support ANSI
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# Disable colors if not supported
if not _supports_color():
    _RESET = _BOLD = _DIM = ""
    _GREEN = _YELLOW = _RED = _CYAN = _MAGENTA = _WHITE = ""
    _BG_GREEN = _BG_RED = _BG_YELLOW = ""


@dataclass
class TaskStat:
    """Metrics for a single task execution."""
    task_id: str
    route: Route
    scored_tokens: int
    total_tokens: int
    latency_ms: float
    fallback_triggered: bool
    confidence: float
    token_budget: int = 0
    cached: bool = False


@dataclass
class TrackerSummary:
    """Aggregate statistics across all tasks."""
    total_tasks: int = 0
    local_tasks: int = 0
    remote_tasks: int = 0
    fallback_tasks: int = 0
    cached_tasks: int = 0
    total_scored_tokens: int = 0
    total_local_tokens: int = 0
    total_remote_tokens: int = 0
    avg_latency_ms: float = 0.0
    avg_confidence_local: float = 0.0
    cache_hit_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_tasks": self.total_tasks,
            "local_tasks": self.local_tasks,
            "remote_tasks": self.remote_tasks,
            "fallback_tasks": self.fallback_tasks,
            "cached_tasks": self.cached_tasks,
            "local_pct": (
                f"{self.local_tasks / self.total_tasks * 100:.1f}%"
                if self.total_tasks else "N/A"
            ),
            "total_scored_tokens": self.total_scored_tokens,
            "total_local_tokens_free": self.total_local_tokens,
            "total_remote_tokens_counted": self.total_remote_tokens,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_confidence_local": round(self.avg_confidence_local, 3),
            "cache_hit_rate": f"{self.cache_hit_rate * 100:.1f}%",
        }


class TokenTracker:
    """Thread-safe tracker for task execution tokens and statistics."""
    
    def __init__(self, log_path: str = "/output/token_log.json") -> None:
        self._lock = threading.Lock()
        self.log_path = log_path
        self.tasks: list[dict] = []
        self._stats: list[TaskStat] = []
        self._recorded_task_ids: set[str] = set()
        
        # Running totals
        self.total_tasks = 0
        self.total_fireworks_tokens = 0
        self.total_free_resolutions = 0
        self.cache_hit_count = 0
        self.rule_based_count = 0
        self.local_model_count = 0
        self.per_model_breakdown: dict[str, int] = {}
        self.per_category_breakdown: dict[str, int] = {}

    def log_fireworks_call(
        self,
        task_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        category: str,
        difficulty: str,
        baseline_tokens: int,
        fallback_triggered: bool = False
    ) -> None:
        """Log a Fireworks API remote model call thread-safely."""
        total_tokens = prompt_tokens + completion_tokens
        with self._lock:
            if task_id in self._recorded_task_ids:
                return
            self._recorded_task_ids.add(task_id)
            
            self.total_tasks += 1
            self.total_fireworks_tokens += total_tokens
            self.per_model_breakdown[model] = self.per_model_breakdown.get(model, 0) + total_tokens
            self.per_category_breakdown[category] = self.per_category_breakdown.get(category, 0) + total_tokens
            
            # Map to legacy TaskStat
            stat = TaskStat(
                task_id=task_id,
                route=Route.REMOTE,
                scored_tokens=total_tokens,
                total_tokens=total_tokens,
                latency_ms=0.0,
                fallback_triggered=fallback_triggered,
                confidence=1.0,
                token_budget=baseline_tokens,
                cached=False
            )
            self._stats.append(stat)

            log_entry = {
                "task_id": task_id,
                "type": "fireworks",
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "baseline_tokens": baseline_tokens,
                "timestamp": time.time(),
                "category": category,
                "difficulty": difficulty,
                "fallback_triggered": fallback_triggered
            }
            self.tasks.append(log_entry)
            self._persist()

    def log_free_resolution(
        self,
        task_id: str,
        resolution_type: str,
        reason: str,
        category: str,
        difficulty: str,
        baseline_tokens: int
    ) -> None:
        """Log a free local/rule/cached resolution thread-safely."""
        with self._lock:
            if task_id in self._recorded_task_ids:
                return
            self._recorded_task_ids.add(task_id)
            
            self.total_tasks += 1
            self.total_free_resolutions += 1
            
            cached = False
            if resolution_type == "cache hit":
                self.cache_hit_count += 1
                cached = True
            elif resolution_type == "rule-based":
                self.rule_based_count += 1
            elif resolution_type == "local model":
                self.local_model_count += 1
                
            # Map to legacy TaskStat
            stat = TaskStat(
                task_id=task_id,
                route=Route.LOCAL,
                scored_tokens=0,
                total_tokens=0,
                latency_ms=0.0,
                fallback_triggered=False,
                confidence=1.0,
                token_budget=baseline_tokens,
                cached=cached
            )
            self._stats.append(stat)

            log_entry = {
                "task_id": task_id,
                "type": "free",
                "resolution_type": resolution_type,
                "reason": reason,
                "baseline_tokens": baseline_tokens,
                "timestamp": time.time(),
                "category": category,
                "difficulty": difficulty,
                "fallback_triggered": False
            }
            self.tasks.append(log_entry)
            self._persist()

    def _persist(self) -> None:
        """Write current stats and logs to the JSON log path."""
        try:
            path = Path(self.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "totals": {
                    "total_tasks": self.total_tasks,
                    "total_fireworks_tokens": self.total_fireworks_tokens,
                    "total_free_resolutions": self.total_free_resolutions,
                    "cache_hit_count": self.cache_hit_count,
                    "rule_based_count": self.rule_based_count,
                    "local_model_count": self.local_model_count,
                    "per_model_breakdown": self.per_model_breakdown,
                    "per_category_breakdown": self.per_category_breakdown
                },
                "log": self.tasks
            }, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("Could not write token log: %s", e)

    def report(self) -> None:
        """Prints a clean summary table of token usage to stderr/logs."""
        import sys
        with self._lock:
            total_tasks = self.total_tasks
            total_tokens = self.total_fireworks_tokens
            free_resolutions = self.total_free_resolutions
            
            # Baseline is if we had executed the largest model (gpt-oss-120b) on every task
            total_baseline = sum(t.get("baseline_tokens", 250) for t in self.tasks)
            tokens_saved = max(0, total_baseline - total_tokens)
            savings_pct = (tokens_saved / total_baseline * 100) if total_baseline else 0.0
            free_pct = (free_resolutions / total_tasks * 100) if total_tasks else 0.0
            
            # Header
            sys.stderr.write("\n" + "=" * 65 + "\n")
            sys.stderr.write("             AMD HACKATHON ROUTER REPORT (TERA TRACK 1)             \n")
            sys.stderr.write("=" * 65 + "\n")
            sys.stderr.write(f"Total Tasks Processed:       {total_tasks}\n")
            sys.stderr.write(f"Total Fireworks Tokens:      {total_tokens}\n")
            sys.stderr.write(f"Estimated Baseline (120B):   {total_baseline}\n")
            sys.stderr.write(f"Tokens Saved (vs 120B):      {tokens_saved} ({savings_pct:.1f}% savings)\n")
            sys.stderr.write(f"Free Resolutions (0 token):  {free_resolutions} ({free_pct:.1f}% of tasks)\n")
            sys.stderr.write(f"  - Cache Hits:              {self.cache_hit_count}\n")
            sys.stderr.write(f"  - Rule-based:              {self.rule_based_count}\n")
            sys.stderr.write(f"  - Local LLM:               {self.local_model_count}\n")
            sys.stderr.write("-" * 65 + "\n")
            
            sys.stderr.write("Breakdown by Model:\n")
            if self.per_model_breakdown:
                for model, tokens in self.per_model_breakdown.items():
                    sys.stderr.write(f"  - {model:<40}: {tokens} tokens\n")
            else:
                sys.stderr.write("  - No Fireworks models called\n")
                
            sys.stderr.write("Breakdown by Category:\n")
            if self.per_category_breakdown:
                for cat, tokens in self.per_category_breakdown.items():
                    sys.stderr.write(f"  - {cat:<40}: {tokens} tokens\n")
            else:
                sys.stderr.write("  - No tokens charged per category\n")
            sys.stderr.write("=" * 65 + "\n\n")
            sys.stderr.flush()


class UsageTracker(TokenTracker):
    """Wrapper class for backward compatibility with existing evaluate/main endpoints."""
    
    def __init__(self, log_path: str = "/output/token_log.json") -> None:
        super().__init__(log_path=log_path)

    def record(self, response: AgentResponse, token_budget: int = 0, cached: bool = False) -> None:
        category = response.routing.signals.get("category", "general")
        difficulty = response.routing.difficulty.value
        baseline_tokens = 150 + token_budget

        with self._lock:
            if response.task_id in self._recorded_task_ids:
                # Update existing task stat with accurate latency/confidence
                stat = None
                for s in self._stats:
                    if s.task_id == response.task_id:
                        stat = s
                        break
                if stat:
                    stat.latency_ms = response.result.latency_ms
                    stat.confidence = response.result.confidence
                    
                    route_color = _GREEN if stat.route == Route.LOCAL else _YELLOW
                    cache_tag = f" {_CYAN}[CACHED]{_RESET}" if stat.cached else ""
                    fallback_tag = f" {_RED}[FALLBACK]{_RESET}" if stat.fallback_triggered else ""
                    print(
                        f"  {_DIM}{'>'}{_RESET} {_WHITE}{stat.task_id:<16}{_RESET} "
                        f"{route_color}{stat.route.value:<8}{_RESET} "
                        f"scored={_BOLD}{stat.scored_tokens:<6}{_RESET} "
                        f"latency={stat.latency_ms:>7.0f}ms "
                        f"conf={stat.confidence:.2f}"
                        f"{cache_tag}{fallback_tag}",
                        flush=True,
                    )
                return

            self._recorded_task_ids.add(response.task_id)

            # Write to legacy TaskStat
            stat = TaskStat(
                task_id=response.task_id,
                route=response.result.route_used,
                scored_tokens=response.scored_tokens,
                total_tokens=response.result.token_usage.total_tokens,
                latency_ms=response.result.latency_ms,
                fallback_triggered=response.result.fallback_triggered,
                confidence=response.result.confidence,
                token_budget=token_budget,
                cached=cached,
            )
            self._stats.append(stat)

        # Print live task log to stdout
        route_color = _GREEN if stat.route == Route.LOCAL else _YELLOW
        cache_tag = f" {_CYAN}[CACHED]{_RESET}" if stat.cached else ""
        fallback_tag = f" {_RED}[FALLBACK]{_RESET}" if stat.fallback_triggered else ""
        print(
            f"  {_DIM}{'>'}{_RESET} {_WHITE}{stat.task_id:<16}{_RESET} "
            f"{route_color}{stat.route.value:<8}{_RESET} "
            f"scored={_BOLD}{stat.scored_tokens:<6}{_RESET} "
            f"latency={stat.latency_ms:>7.0f}ms "
            f"conf={stat.confidence:.2f}"
            f"{cache_tag}{fallback_tag}",
            flush=True,
        )

        if cached:
            self.log_free_resolution(
                task_id=response.task_id,
                resolution_type="cache hit",
                reason="Normalized prompt cache hit",
                category=category,
                difficulty=difficulty,
                baseline_tokens=baseline_tokens
            )
        elif response.result.route_used == Route.LOCAL:
            is_rule = response.result.latency_ms < 10.0 and response.result.confidence == 1.0
            res_type = "rule-based" if is_rule else "local model"
            reason = "Zero-cost rule matcher" if is_rule else "Resolved by local model"
            self.log_free_resolution(
                task_id=response.task_id,
                resolution_type=res_type,
                reason=reason,
                category=category,
                difficulty=difficulty,
                baseline_tokens=baseline_tokens
            )
        else:
            model_name = response.result.route_used.value
            self.log_fireworks_call(
                task_id=response.task_id,
                model=model_name,
                prompt_tokens=response.result.token_usage.prompt_tokens,
                completion_tokens=response.result.token_usage.completion_tokens,
                category=category,
                difficulty=difficulty,
                baseline_tokens=baseline_tokens,
                fallback_triggered=response.result.fallback_triggered
            )

    def summarize(self, cache_hit_rate: float = 0.0) -> TrackerSummary:
        """Compute aggregate statistics for evaluation harness."""
        with self._lock:
            if not self._stats:
                return TrackerSummary()

            local_stats = [s for s in self._stats if s.route == Route.LOCAL]
            remote_stats = [s for s in self._stats if s.route == Route.REMOTE]
            fallback_stats = [s for s in self._stats if s.fallback_triggered]
            cached_stats = [s for s in self._stats if s.cached]

            total_latency = sum(s.latency_ms for s in self._stats)
            local_confidences = [
                s.confidence for s in local_stats if s.confidence > 0
            ]

            return TrackerSummary(
                total_tasks=len(self._stats),
                local_tasks=len(local_stats),
                remote_tasks=len(remote_stats),
                fallback_tasks=len(fallback_stats),
                cached_tasks=len(cached_stats),
                total_scored_tokens=sum(s.scored_tokens for s in self._stats),
                total_local_tokens=sum(s.total_tokens for s in local_stats),
                total_remote_tokens=sum(s.total_tokens for s in remote_stats),
                avg_latency_ms=total_latency / len(self._stats),
                avg_confidence_local=(
                    sum(local_confidences) / len(local_confidences)
                    if local_confidences
                    else 0.0
                ),
                cache_hit_rate=cache_hit_rate,
            )

    def print_report(self, cache_hit_rate: float = 0.0) -> None:
        """Print the colored execution report to stdout."""
        summary = self.summarize(cache_hit_rate)
        data = summary.to_dict()

        # Token savings estimate (vs all-remote baseline)
        baseline_estimate = summary.total_tasks * 200
        saved = baseline_estimate - summary.total_scored_tokens
        savings_pct = (saved / baseline_estimate * 100) if baseline_estimate else 0

        print()
        print(f"  {_BOLD}{_CYAN}{'=' * 58}{_RESET}")
        print(f"  {_BOLD}{_WHITE}  HYBRID ROUTING AGENT — EXECUTION REPORT{_RESET}")
        print(f"  {_BOLD}{_CYAN}{'=' * 58}{_RESET}")
        print()

        # Routing distribution bar
        local_pct = summary.local_tasks / summary.total_tasks * 100 if summary.total_tasks else 0
        remote_pct = 100 - local_pct
        bar_width = 40
        local_bar = int(bar_width * local_pct / 100)
        remote_bar = bar_width - local_bar

        print(f"  {_BOLD}Routing Distribution{_RESET}")
        print(
            f"  {_BG_GREEN}{_WHITE}{' ' * local_bar}{_RESET}"
            f"{_BG_RED}{_WHITE}{' ' * remote_bar}{_RESET}"
            f"  {_GREEN}LOCAL {local_pct:.0f}%{_RESET} | "
            f"{_YELLOW}REMOTE {remote_pct:.0f}%{_RESET}"
        )
        print()

        # Key metrics
        metrics = [
            ("Total Tasks", str(summary.total_tasks), _WHITE),
            ("Local (free)", str(summary.local_tasks), _GREEN),
            ("Remote (counted)", str(summary.remote_tasks), _YELLOW),
            ("Fallbacks", str(summary.fallback_tasks), _RED),
            ("Cache Hits", str(summary.cached_tasks), _CYAN),
            ("", "", ""),  # spacer
            ("Scored Tokens", str(summary.total_scored_tokens), _BOLD + _WHITE),
            ("Free Local Tokens", str(summary.total_local_tokens), _GREEN),
            ("Estimated Savings", f"~{savings_pct:.0f}% vs all-remote", _GREEN),
            ("", "", ""),  # spacer
            ("Avg Latency", f"{summary.avg_latency_ms:.0f}ms", _WHITE),
            ("Cache Hit Rate", data["cache_hit_rate"], _CYAN),
        ]

        for label, value, color in metrics:
            if not label:
                print()
                continue
            print(f"  {_DIM}{label:<28}{_RESET} {color}{value}{_RESET}")

        print()
        print(f"  {_BOLD}{_CYAN}{'=' * 58}{_RESET}")
        print()
        
        # Also print the TERA token tracker summary report to stderr/logs
        self.report()

    def export_json(self) -> str:
        """Export all per-task stats as JSON."""
        with self._lock:
            return json.dumps(
                [
                    {
                        "task_id": s.task_id,
                        "route": s.route.value,
                        "scored_tokens": s.scored_tokens,
                        "total_tokens": s.total_tokens,
                        "latency_ms": round(s.latency_ms, 1),
                        "fallback_triggered": s.fallback_triggered,
                        "confidence": round(s.confidence, 3),
                        "token_budget": s.token_budget,
                        "cached": s.cached,
                    }
                    for s in self._stats
                ],
                indent=2,
            )
