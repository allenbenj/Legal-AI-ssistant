"""Core protocol and type definitions for the workflow engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, Sequence, TypeVar

T = TypeVar("T")
T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")


@dataclass
class WorkflowContext:
    """Context passed to workflow nodes during execution."""

    metadata: dict[str, Any] = field(default_factory=dict)


class RetryStrategy(Protocol):
    """Defines retry behavior for a workflow node."""

    async def should_retry(
        self, attempt: int, exc: BaseException, context: WorkflowContext
    ) -> float | None:
        """Return delay before next attempt or ``None`` to stop retrying."""


class MergeStrategy(Protocol, Generic[T]):
    """Strategy for merging results from parallel nodes."""

    async def merge(self, results: Sequence[T]) -> T:
        """Merge a sequence of results into a single output."""


class LegalWorkflowNode(Protocol, Generic[T_In, T_Out]):
    """Protocol that all workflow nodes must implement."""

    async def run(self, data: T_In, context: WorkflowContext) -> T_Out:
        """Process input ``data`` and return an output."""

