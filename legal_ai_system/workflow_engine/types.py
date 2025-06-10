from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar

T = TypeVar("T")
T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")

LegalWorkflowNode = Callable[[T_In], Awaitable[T_Out]]


@dataclass
class WorkflowContext(Generic[T]):
    """Container for workflow state."""

    data: T


MergeStrategy = Callable[[list[Any]], Any]
RetryStrategy = Callable[[LegalWorkflowNode[T_In, T_Out]], Awaitable[T_Out]]

