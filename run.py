"""
Container entry point for the AMD Hackathon submission.

Reads tasks from /input/tasks.json, processes them through the
optimized hybrid agent, and writes results to /output/results.json.

This is the ENTRYPOINT for the Docker container used in evaluation.
For local development, use main.py instead.

Container contract:
  - Input:  /input/tasks.json  → [{"task_id": "t1", "prompt": "..."}, ...]
  - Output: /output/results.json → [{"task_id": "t1", "answer": "..."}, ...]
  - Exit code 0 on success, non-zero on failure
  - Maximum runtime: 10 minutes
  - Env vars FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS
    are injected by the evaluation harness at runtime.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_PATH = os.getenv("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/output/results.json")
MAX_RUNTIME_SECONDS = 570  # 9.5 min safety margin (limit is 10 min)


def setup_logging() -> None:
    """Configure structured logging."""
    level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,  # Keep stdout clean for output
    )


def log_peak_memory(logger) -> None:
    """Read and log peak container memory usage from cgroups or local maxrss (Linux)."""
    for path in ["/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory.max_usage_in_bytes"]:
        try:
            p = Path(path)
            if p.exists():
                bytes_used = int(p.read_text().strip())
                mb_used = bytes_used / (1024 * 1024)
                logger.info("Peak Container Memory Usage (cgroup): %.1f MB", mb_used)
                return
        except Exception:
            pass

    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        mb_used = usage.ru_maxrss / 1024.0
        logger.info("Peak Agent Process Memory Usage (maxrss): %.1f MB", mb_used)
    except Exception:
        pass


def main() -> int:
    """Main container entrypoint. Returns exit code."""
    setup_logging()
    logger = logging.getLogger("run")

    run_start_time = time.time()
    try:
        from agent.governor import set_start_time
        set_start_time(run_start_time)
    except Exception as exc:
        logger.warning("Could not set governor start time: %s", exc)

    start_time = time.monotonic()

    # ── Step 0: Proactively start and verify local Ollama server ──
    try:
        from agent.router import _is_ollama_available
        logger.info("Initializing in-container Ollama check/startup...")
        ollama_start_start = time.monotonic()
        ollama_active = _is_ollama_available()
        ollama_startup_duration = time.monotonic() - ollama_start_start
        logger.info(
            "Local Ollama initialized. Active: %s (Time taken: %.1fs)",
            ollama_active, ollama_startup_duration
        )
    except Exception as exc:
        logger.warning("Could not auto-start local Ollama: %s", exc)

    # ── Step 1: Load configuration ──
    try:
        from agent.config import load_config, get_resolved_model
        from agent.executor import HybridExecutor
        from agent.models import Task
        from agent.router import HeuristicRouter
        from agent.budget import estimate_token_budget

        config = load_config()
        resolved_model = get_resolved_model(config)
        logger.info("Config loaded. Model: %s", resolved_model)
        logger.info("Base URL: %s", config.fireworks.base_url)
        logger.info("ALLOWED_MODELS: %s", config.fireworks.allowed_models or "(not set)")
        logger.info("Cache: %s | Compression: %s",
                     config.cache_enabled, config.compression_enabled)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        return 1

    # ── Step 2: Read input tasks ──
    try:
        input_path = Path(INPUT_PATH)
        logger.info("Reading tasks from %s", input_path)

        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            return 1

        raw = json.loads(input_path.read_text(encoding="utf-8"))

        # Normalize task format: support both "task_id" and "id" keys
        tasks = []
        for item in raw:
            task_id = item.get("task_id") or item.get("id", "")
            prompt = item.get("prompt", "")
            if task_id and prompt:
                tasks.append(Task(id=str(task_id), prompt=prompt))

        logger.info("Loaded %d tasks", len(tasks))
        if not tasks:
            logger.error("No valid tasks found in input file")
            return 1

    except Exception as exc:
        logger.error("Failed to read input tasks: %s", exc)
        return 1

    # ── Step 3: Initialize pipeline ──
    try:
        from agent.tracker import UsageTracker
        tracker = UsageTracker(log_path=os.getenv("TOKEN_LOG_PATH", "/output/token_log.json"))
        router = HeuristicRouter(config.router)
        executor = HybridExecutor(config, tracker=tracker)
    except Exception as exc:
        logger.error("Failed to initialize pipeline: %s", exc)
        return 1

    # ── Step 4: Process tasks ──
    results = []
    for i, task in enumerate(tasks):
        # Safety: check runtime limit
        elapsed = time.monotonic() - start_time
        if elapsed > MAX_RUNTIME_SECONDS:
            logger.warning(
                "Approaching 10-min runtime limit (%.0fs elapsed). "
                "Stopping with %d/%d tasks completed.",
                elapsed, i, len(tasks),
            )
            break

        task_start = time.monotonic()
        try:
            decision = router.route(task)
            result = executor.execute(task, decision)
            answer = result.output

            task_duration = time.monotonic() - task_start
            logger.info(
                "Task %s: route=%s tokens=%d latency=%.0fms (total duration=%.1fs) category=%s",
                task.id,
                result.route_used.value,
                result.token_usage.total_tokens,
                result.latency_ms,
                task_duration,
                decision.difficulty.value,
            )
            if task_duration > 30.0:
                logger.warning("Task %s exceeded 30s limit (duration=%.1fs)!", task.id, task_duration)
        except Exception as exc:
            task_duration = time.monotonic() - task_start
            logger.error("Task %s failed in %.1fs: %s", task.id, task_duration, exc)
            answer = ""  # Empty answer is better than crashing

        results.append({
            "task_id": task.id,
            "answer": answer,
        })

    # ── Step 5: Write output ──
    try:
        output_path = Path(OUTPUT_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Results written to %s (%d tasks)", output_path, len(results))
    except Exception as exc:
        logger.error("Failed to write output: %s", exc)
        return 1

    # ── Summary ──
    total_elapsed = time.monotonic() - start_time
    if hasattr(executor, 'cache'):
        logger.info("Cache hits: %d", executor.cache.stats.hits)
    logger.info("Total runtime: %.1fs | Tasks: %d", total_elapsed, len(results))
    
    try:
        tracker.report()
    except Exception as exc:
        logger.error("Failed to print token report: %s", exc)

    # Log peak memory usage stats
    log_peak_memory(logger)

    return 0


if __name__ == "__main__":
    sys.exit(main())
