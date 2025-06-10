from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Iterable, List, Tuple

from .merge import ListMerge, MergeStrategy
from .retry import ExponentialBackoffRetry


class LegalWorkflowBuilder:
    """Simple async workflow builder with optional parallel steps."""

    def __init__(self) -> None:
        self._steps: List[
            Callable[[Any], Awaitable[Any]]
            | Tuple[List[Callable[[Any], Awaitable[Any]]], MergeStrategy]
        ] = []

    def add_step(self, func: Callable[[Any], Awaitable[Any]]) -> None:
        """Append a sequential processing step."""

        self._steps.append(func)

    def add_parallel_processing(
        self,
        funcs: Iterable[Callable[[Any], Awaitable[Any]]],
        *,
        merge_strategy: MergeStrategy | None = None,
    ) -> None:
        """Add a parallel processing block.

        Each function is executed concurrently and the results are combined
        using ``merge_strategy``.  If no strategy is provided ``ListMerge`` is
        used.
        """

        strategy = merge_strategy or ListMerge()
        self._steps.append((list(funcs), strategy))

    async def run(self, data: Any) -> Any:
        """Execute the workflow returning the final result."""

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
