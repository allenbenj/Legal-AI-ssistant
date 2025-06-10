from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, List

from .merge import MergeStrategy, ListMerge
from .retry import ExponentialBackoffRetry

Step = Callable[[Any], Awaitable[Any]]


@dataclass
class _ParallelStep:
    nodes: List[Step]
    strategy: MergeStrategy


class LegalWorkflowBuilder:
    """Minimal async workflow builder for composing async callables."""

    def __init__(self, retry: ExponentialBackoffRetry | None = None) -> None:
        self._steps: List[Step | _ParallelStep] = []
        self._retry = retry or ExponentialBackoffRetry()

    def add_step(self, node: Step) -> None:
        """Append a sequential processing node."""
        self._steps.append(node)

    def add_parallel_processing(
        self,
        nodes: Iterable[Step],
        *,
        merge_strategy: MergeStrategy | None = None,
    ) -> None:
        """Execute ``nodes`` concurrently and merge their results."""
        strategy = merge_strategy or ListMerge()
        self._steps.append(_ParallelStep(list(nodes), strategy))

    async def run(self, data: Any) -> Any:
        """Run the configured workflow."""
        result: Any = data
        for step in self._steps:
            if isinstance(step, _ParallelStep):
                tasks = [self._retry.run(n, result) for n in step.nodes]
                results = await asyncio.gather(*tasks)
                result = step.strategy.merge(results)
            else:
                result = await self._retry.run(step, result)
        return result


__all__ = ["LegalWorkflowBuilder"]
