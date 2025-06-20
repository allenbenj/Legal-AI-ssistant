from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Iterable, List, Tuple

from .merge import ConcatMerge, MergeStrategy
from .retry import ExponentialBackoffRetry

LegalWorkflowNode = Callable[[Any], Awaitable[Any]]


class LegalWorkflowBuilder:
    """Asynchronous workflow builder with optional parallel steps."""

    def __init__(self) -> None:
        self._steps: List[LegalWorkflowNode | Tuple[List[LegalWorkflowNode], MergeStrategy]] = []

    def add_step(self, func: LegalWorkflowNode) -> None:
        """Append a sequential processing step."""
        self._steps.append(func)

    def add_parallel_processing(
        self,
        funcs: Iterable[LegalWorkflowNode],
        *,
        merge_strategy: MergeStrategy | None = None,
    ) -> None:
        """Execute ``funcs`` concurrently and merge results using ``merge_strategy``."""
        strategy = merge_strategy or ConcatMerge()
        self._steps.append((list(funcs), strategy))

    async def run(self, data: Any) -> Any:
        """Run the workflow sequentially with automatic retries."""
        retry = ExponentialBackoffRetry()
        result = data
        for step in self._steps:
            if isinstance(step, tuple):
                funcs, strategy = step
                results = await asyncio.gather(
                    *[retry.run(f, result) for f in funcs]
                )
                result = strategy.merge(list(results))
            else:
                result = await retry.run(step, result)
        return result


__all__ = ["LegalWorkflowBuilder"]

