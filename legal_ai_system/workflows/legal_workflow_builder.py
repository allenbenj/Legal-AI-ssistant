from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Iterable, List

from .merge import MergeStrategy, ListMerge
from .retry import ExponentialBackoffRetry


class LegalWorkflowBuilder:
    """Simple async workflow builder supporting sequential and parallel steps."""

    def __init__(self) -> None:
        self._steps: List[Callable[[Any], Awaitable[Any]]] = []

    def add_step(self, step: Callable[[Any], Awaitable[Any]]) -> "LegalWorkflowBuilder":
        """Append a sequential step to the workflow."""
        self._steps.append(step)
        return self

    def add_parallel_processing(
        self,
        nodes: Iterable[Callable[[Any], Awaitable[Any]]],
        merge_strategy: MergeStrategy | None = None,
    ) -> "LegalWorkflowBuilder":
        """Add a parallel processing stage.

        Each node is executed concurrently and results are merged using
        ``merge_strategy``. If a node fails, it is retried using
        :class:`ExponentialBackoffRetry`.
        """

        strategy = merge_strategy or ListMerge()
        node_list = list(nodes)
        retry = ExponentialBackoffRetry()

        async def parallel_step(data: Any) -> Any:
            results = await asyncio.gather(
                *[retry.run(node, data) for node in node_list],
                return_exceptions=True,
            )
            # Propagate the first exception if all retries failed for a node
            for res in results:
                if isinstance(res, Exception):
                    raise res
            return strategy.merge(results)  # type: ignore[arg-type]

        self._steps.append(parallel_step)
        return self

    async def run(self, data: Any) -> Any:
        """Execute the workflow asynchronously."""
        result = data
        for step in self._steps:
            result = await step(result)
        return result


__all__ = ["LegalWorkflowBuilder"]
