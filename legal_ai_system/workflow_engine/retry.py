"""Retry strategies for workflow nodes."""

from __future__ import annotations

from dataclasses import dataclass

from .types import RetryStrategy, WorkflowContext


@dataclass
class ExponentialBackoffRetry(RetryStrategy):
    """Exponential backoff retry strategy."""

    max_attempts: int = 3
    base_delay: float = 1.0

    async def should_retry(
        self, attempt: int, exc: BaseException, context: WorkflowContext
    ) -> float | None:
        if attempt >= self.max_attempts:
            return None
        return self.base_delay * 2 ** (attempt - 1)
