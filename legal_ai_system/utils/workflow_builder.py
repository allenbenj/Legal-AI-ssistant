from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple, Awaitable


class WorkflowBuilder:
    """Simple asynchronous workflow builder supporting branching and merges."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Callable[[Any], Awaitable[Any]]] = {}
        self._entry: Optional[str] = None
        self._transitions: Dict[str, List[Tuple[str, Optional[Callable[[Any], bool]], bool]]] = {}
        self._parallel: Dict[str, Tuple[List[str], str]] = {}

    def register_node(self, name: str, func: Callable[[Any], Awaitable[Any]]) -> None:
        self._nodes[name] = func

    def set_entry_point(self, name: str) -> None:
        self._entry = name

    def add_edge(
        self,
        start: str,
        end: str,
        condition: Optional[Callable[[Any], bool]] = None,
        parallel: bool = False,
    ) -> None:
        self._transitions.setdefault(start, []).append((end, condition, parallel))

    def add_parallel(self, start: str, branches: List[str], merge: str) -> None:
        self._parallel[start] = (branches, merge)

    async def run(self, data: Any) -> Any:
        if self._entry is None:
            raise RuntimeError("Entry point not set")
        return await self._execute(self._entry, data)

    async def _execute(self, node_name: str, data: Any) -> Any:
        node = self._nodes[node_name]
        result = await node(data)

        if node_name in self._parallel:
            branches, merge_node = self._parallel[node_name]
            branch_results = await asyncio.gather(
                *[self._execute(b, result) for b in branches]
            )
            result = await self._nodes[merge_node](branch_results)
            node_name = merge_node

        for next_node, cond, parallel in self._transitions.get(node_name, []):
            if cond is None or cond(result):
                if parallel:
                    asyncio.create_task(self._execute(next_node, result))
                else:
                    result = await self._execute(next_node, result)
        return result

__all__ = ["WorkflowBuilder"]
