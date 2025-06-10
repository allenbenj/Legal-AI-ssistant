"""Core typing utilities for the workflow engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Protocol, Sequence, TypeVar, Generic, Callable

T = TypeVar("T")
T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")


@dataclass
class WorkflowContext:
    """Container for passing metadata and shared state between nodes."""

    data: dict[str, Any] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RetryStrategy(Protocol):
    """Protocol for retrying operations."""

    async def execute(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """Execute ``func`` with retry semantics and return its result."""


class MergeStrategy(Protocol, Generic[T]):
    """Protocol for merging multiple results into one."""

    async def merge(self, results: Sequence[T]) -> T:
        """Merge a sequence of ``results`` and return a single value."""


class LegalWorkflowNode(Protocol, Generic[T_In, T_Out]):
    """Unit of work executed within a workflow."""

    async def run(self, data: T_In, context: WorkflowContext) -> T_Out:
        """Process ``data`` returning an output."""
