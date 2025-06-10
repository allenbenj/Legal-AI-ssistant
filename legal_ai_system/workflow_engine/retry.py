"""Retry strategies for workflow nodes."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, TypeVar

from .types import RetryStrategy

T = TypeVar("T")


class ExponentialBackoffRetry(RetryStrategy):
    """Retry strategy using exponential backoff."""

    def __init__(self, attempts: int = 3, base_delay: float = 0.5) -> None:
        self.attempts = attempts
        self.base_delay = base_delay

    async def execute(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        last_exc: BaseException | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - propagate after retries
                last_exc = exc
                if attempt == self.attempts:
                    break
                delay = self.base_delay * 2 ** (attempt - 1)
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc
