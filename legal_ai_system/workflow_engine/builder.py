"""Composable workflow builder."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, List, Sequence

from .merge import ConcatMerge
from .types import (
    LegalWorkflowNode,
    MergeStrategy,
    RetryStrategy,
    WorkflowContext,
)


@dataclass
class _SequentialNode(LegalWorkflowNode[Any, Any]):
    nodes: Sequence[LegalWorkflowNode[Any, Any]]

    async def run(self, data: Any, context: WorkflowContext) -> Any:
        result: Any = data
        for node in self.nodes:
            result = await node.run(result, context)
        return result


@dataclass
class _RetryNode(LegalWorkflowNode[Any, Any]):
    node: LegalWorkflowNode[Any, Any]
    strategy: RetryStrategy

    async def run(self, data: Any, context: WorkflowContext) -> Any:
        attempt = 0
        while True:
            try:
                return await self.node.run(data, context)
            except Exception as exc:  # pragma: no cover - simple retry wrapper
                attempt += 1
                delay = await self.strategy.should_retry(attempt, exc, context)
                if delay is None:
                    raise
                await asyncio.sleep(delay)


@dataclass
class _ParallelNode(LegalWorkflowNode[Any, Any]):
    nodes: Sequence[LegalWorkflowNode[Any, Any]]
    merge: MergeStrategy[Any] = ConcatMerge()

    async def run(self, data: Any, context: WorkflowContext) -> Any:
        results = await asyncio.gather(
            *(node.run(data, context) for node in self.nodes)
        )
        return await self.merge.merge(results)


@dataclass
class _BranchNode(LegalWorkflowNode[Any, Any]):
    condition: Callable[[Any, WorkflowContext], Awaitable[bool]]
    if_true: LegalWorkflowNode[Any, Any]
    if_false: LegalWorkflowNode[Any, Any] | None = None

    async def run(self, data: Any, context: WorkflowContext) -> Any:
        if await self.condition(data, context):
            return await self.if_true.run(data, context)
        if self.if_false:
            return await self.if_false.run(data, context)
        return data


class LegalWorkflowBuilder:
    """Utility to compose complex workflows."""

    def __init__(self) -> None:
        self._nodes: List[LegalWorkflowNode[Any, Any]] = []

    def add_node(
        self,
        node: LegalWorkflowNode[Any, Any],
        *,
        retry: RetryStrategy | None = None,
    ) -> "LegalWorkflowBuilder":
        if retry:
            node = _RetryNode(node=node, strategy=retry)
        self._nodes.append(node)
        return self

    def add_conditional(
        self,
        condition: Callable[[Any, WorkflowContext], Awaitable[bool]],
        true_branch: "LegalWorkflowBuilder",
        false_branch: "LegalWorkflowBuilder" | None = None,
    ) -> "LegalWorkflowBuilder":
        branch_node = _BranchNode(
            condition=condition,
            if_true=true_branch.build(),
            if_false=false_branch.build() if false_branch else None,
        )
        self._nodes.append(branch_node)
        return self

    def add_parallel(
        self,
        nodes: Iterable[LegalWorkflowNode[Any, Any]],
        *,
        merge: MergeStrategy[Any] | None = None,
    ) -> "LegalWorkflowBuilder":
        parallel_node = _ParallelNode(
            nodes=list(nodes),
            merge=merge or ConcatMerge(),
        )
        self._nodes.append(parallel_node)
        return self

    def build(self) -> LegalWorkflowNode[Any, Any]:
        return _SequentialNode(self._nodes)

    async def run(self, data: Any, context: WorkflowContext | None = None) -> Any:
        ctx = context or WorkflowContext()
        return await self.build().run(data, ctx)
