"""Composable workflow builder supporting branches and parallel execution."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Iterable, List, Optional

from .merge import ConcatMerge
from .types import (
    LegalWorkflowNode,
    MergeStrategy,
    RetryStrategy,
    WorkflowContext,
)


class LegalWorkflowBuilder:
    """Build and execute asynchronous workflows."""

    def __init__(
        self,
        retry_strategy: RetryStrategy | None = None,
    ) -> None:
        self._steps: List[Callable[[Any, WorkflowContext], Awaitable[Any]]] = []
        self._retry = retry_strategy

    # -----------------------------------------------------
    async def _run_node(
        self, node: LegalWorkflowNode[Any, Any], data: Any, ctx: WorkflowContext
    ) -> Any:
        if self._retry is not None:
            return await self._retry.execute(node.run, data, ctx)
        return await node.run(data, ctx)

    # -----------------------------------------------------
    def add_node(self, node: LegalWorkflowNode[Any, Any]) -> "LegalWorkflowBuilder":
        """Append a node to the workflow."""

        async def step(data: Any, ctx: WorkflowContext) -> Any:
            return await self._run_node(node, data, ctx)

        self._steps.append(step)
        return self

    # -----------------------------------------------------
    def add_conditional(
        self,
        predicate: Callable[[Any, WorkflowContext], Awaitable[bool] | bool],
        true_branch: "LegalWorkflowBuilder",
        false_branch: Optional["LegalWorkflowBuilder"] = None,
    ) -> "LegalWorkflowBuilder":
        """Execute branches based on a predicate."""

        async def step(data: Any, ctx: WorkflowContext) -> Any:
            pred = predicate(data, ctx)
            if asyncio.iscoroutine(pred):
                pred = await pred  # type: ignore[assignment]
            if pred:
                return await true_branch.execute(data, ctx)
            if false_branch is not None:
                return await false_branch.execute(data, ctx)
            return data

        self._steps.append(step)
        return self

    # -----------------------------------------------------
    def add_parallel(
        self,
        nodes: Iterable[LegalWorkflowNode[Any, Any]],
        merge: MergeStrategy | None = None,
    ) -> "LegalWorkflowBuilder":
        """Run a group of nodes concurrently and merge their results."""

        strategy = merge or ConcatMerge()

        async def step(data: Any, ctx: WorkflowContext) -> Any:
            tasks = [self._run_node(node, data, ctx) for node in nodes]
            results = await asyncio.gather(*tasks)
            return await strategy.merge(results)

        self._steps.append(step)
        return self

    # -----------------------------------------------------
    async def execute(
        self, data: Any, context: WorkflowContext | None = None
    ) -> Any:
        """Run the workflow against ``data``."""

        ctx = context or WorkflowContext()
        result: Any = data
        for step in self._steps:
            result = await step(result, ctx)
        return result

