"""
Adaptive Governor for managing container execution time constraints.
"""

import time
import logging

logger = logging.getLogger(__name__)

# Start time epoch for tracking container elapsed time
_START_TIME = time.time()

def set_start_time(t: float) -> None:
    """Set the container startup timestamp."""
    global _START_TIME
    _START_TIME = t

def get_current_sample_count(total_budget_seconds: float = 570.0) -> int:
    """
    Returns the dynamic sample count (3, 2, or 1) based on the percent of budget elapsed.
    """
    try:
        elapsed = time.time() - _START_TIME
        pct_elapsed = elapsed / max(total_budget_seconds, 1.0)
        
        if pct_elapsed < 0.70:
            count = 3
        elif pct_elapsed < 0.85:
            count = 2
        else:
            count = 1
            
        logger.debug(
            "Adaptive Governor: elapsed=%.1fs (%.1f%% of %.1fs budget) -> sample_count=%d",
            elapsed, pct_elapsed * 100.0, total_budget_seconds, count
        )
        return count
    except Exception as exc:
        logger.warning("Error in Adaptive Governor calculation: %s. Falling back to 1 sample.", exc)
        return 1
