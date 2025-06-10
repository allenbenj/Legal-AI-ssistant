from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, List, Generic, TypeVar, Union

from .merge import MergeStrategy, ListMerge
from .retry import ExponentialBackoffRetry

T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")


@dataclass
class _ParallelStep:
    functions: List[Callable[[Any], Awaitable[Any]]]
    merge_strategy: MergeStrategy


class LegalWorkflowBuilder(Generic[T_In, T_Out]):
    """Simple async workflow builder for chaining callables."""

    def __init__(self, retry_strategy: ExponentialBackoffRetry | None = None) -> None:
        self._steps: List[Union[Callable[[Any], Awaitable[Any]], _ParallelStep]] = []
        self._retry_strategy = retry_strategy or ExponentialBackoffRetry()

    def add_step(self, func: Callable[[Any], Awaitable[Any]]) -> None:
        """Add a sequential processing step."""

        self._steps.append(func)

    def add_parallel_processing(
        self,
        funcs: Iterable[Callable[[Any], Awaitable[Any]]],
        merge_strategy: MergeStrategy | None = None,
    ) -> None:
        """Execute callables in parallel and merge the results."""

        strategy = merge_strategy or ListMerge()
        self._steps.append(_ParallelStep(list(funcs), strategy))

    async def run(self, data: T_In) -> T_Out:
        """Run the configured workflow returning the final result."""

        result: Any = data
        for step in self._steps:
            if isinstance(step, _ParallelStep):
                results = await asyncio.gather(
                    *[self._run_with_retry(func, result) for func in step.functions]
                )
                result = step.merge_strategy.merge(list(results))
            else:
                result = await self._run_with_retry(step, result)
        return result  # type: ignore[return-value]

    async def _run_with_retry(
        self, func: Callable[[Any], Awaitable[Any]], arg: Any
    ) -> Any:
        return await self._retry_strategy.run(func, arg)


__all__ = ["LegalWorkflowBuilder"]
