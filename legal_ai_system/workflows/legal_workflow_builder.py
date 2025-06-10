from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, List, Sequence

from ..workflow_engine.merge import ListMerge, MergeStrategy



@dataclass
class _ParallelStep:
    funcs: Sequence[Callable[[Any], Awaitable[Any] | Any]]
    merge_strategy: MergeStrategy

    async def __call__(self, input_data: Any) -> Any:
        results = await asyncio.gather(*(f(input_data) for f in self.funcs))
        return self.merge_strategy.merge(results)


class LegalWorkflowBuilder:
    """Simple async workflow executor for unit tests."""

    def __init__(self) -> None:
        self._steps: List[Callable[[Any], Awaitable[Any] | Any]] = []
        self._error_handler: Callable[[Any], Awaitable[Any]] | None = None


        self._error_handler = handler

    def add_step(self, func: Callable[[Any], Awaitable[Any] | Any]) -> None:
        """Add a sequential processing step."""
        self._steps.append(func)

    def add_parallel_processing(
        self,
        funcs: Iterable[Callable[[Any], Awaitable[Any] | Any]],
        *,
        merge_strategy: MergeStrategy | None = None,
    ) -> None:
        """Add a set of functions to run in parallel and merge their results."""
        strategy = merge_strategy or ListMerge()
        self._steps.append(_ParallelStep(list(funcs), strategy))

    async def run(self, initial_input: Any) -> Any:
        """Execute the workflow and return the final result."""

        data: Any = initial_input
        for step in self._steps:
            result = step(data)
            data = await result if asyncio.iscoroutine(result) else result

            data = await self._error_handler(data)
        return data



__all__ = ["LegalWorkflowBuilder", "MergeStrategy", "ListMerge"]
