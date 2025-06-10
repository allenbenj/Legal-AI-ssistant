from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


class ExponentialBackoffRetry:
    """Retry helper implementing exponential backoff."""

    def __init__(
        self, max_attempts: int = 3, base_delay: float = 0.5, multiplier: float = 2.0
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.multiplier = multiplier

    async def run(self, func: Callable[[Any], Awaitable[T]], *args: Any) -> T:
        attempt = 0
        while True:
            try:
                return await func(*args)
            except Exception:
                attempt += 1
                if attempt >= self.max_attempts:
                    raise
                delay = self.base_delay * (self.multiplier ** (attempt - 1))
                await asyncio.sleep(delay)

__all__ = ["ExponentialBackoffRetry"]

