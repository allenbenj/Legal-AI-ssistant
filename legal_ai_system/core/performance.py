import asyncio
import time
from functools import wraps
from typing import Callable, Dict, List

# Simple tracker storing durations for operations
performance_tracker: Dict[str, List[float]] = {}


def measure_performance(operation: str) -> Callable:
    """Decorator to measure asynchronous function performance."""

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                duration = time.perf_counter() - start
                performance_tracker.setdefault(operation, []).append(duration)
                return result

            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                performance_tracker.setdefault(operation, []).append(duration)
                return result

            return sync_wrapper

    return decorator

__all__ = ["measure_performance", "performance_tracker"]
