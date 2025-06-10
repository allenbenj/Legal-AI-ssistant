from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


class BaseNode:
    """Minimal ``BaseNode`` placeholder used for local development."""

    pass


class StateGraph:
    """Simplified asynchronous graph structure used for workflows."""

    def __init__(self) -> None:
        # registered graph nodes by name
        self._nodes: Dict[str, Callable[[Any], Any]] = {}
        # transition table: start -> list[(end, condition, parallel)]
        self._edges: Dict[str, List[Tuple[str, Optional[Callable[[Any], bool]], bool]]] = {}
        # parallel branch specification: start -> (branches, merge)
        self._parallel: Dict[str, Tuple[List[str], str]] = {}
        self._entry_point: Optional[str] = None

    def add_node(self, name: str, node: object) -> None:
        """Register a node callable."""

        self._nodes[name] = node

    def set_entry_point(self, name: str) -> None:
        self._entry_point = name

    def add_edge(
        self,
        start: str,
        end: str,
        condition: Optional[Callable[[Any], bool]] = None,
        parallel: bool = False,
    ) -> None:
        """Add an edge between two nodes.

        ``condition`` is a callable returning ``bool`` that determines whether
        this transition should be taken. If ``parallel`` is ``True`` the edge is
        executed asynchronously without waiting for completion.
        """

        self._edges.setdefault(start, []).append((end, condition, parallel))

    def add_parallel_nodes(self, start: str, branches: list[str], merge: str) -> None:
        """Execute ``branches`` in parallel before running ``merge``."""

        self._parallel[start] = (branches, merge)

    def add_conditional_edges(self, start: str, pairs: List[Tuple[Callable[[Any], bool], str]]) -> None:
        """Convenience helper for adding multiple conditional transitions."""

        for cond, end in pairs:
            self.add_edge(start, end, condition=cond)

    def run(self, input_text: Any) -> Any:
        """Execute the graph synchronously using ``asyncio`` internally."""

        async def _execute(node_name: str, data: Any) -> Any:
            if node_name == END:
                return data
            node = self._nodes[node_name]
            result = node(data)
            if asyncio.iscoroutine(result):
                result = await result

            if node_name in self._parallel:
                branches, merge_node = self._parallel[node_name]
                branch_results = await asyncio.gather(
                    *[_execute(b, result) for b in branches]
                )
                result = await _execute(merge_node, branch_results)
                node_name = merge_node

            for next_node, cond, parallel in self._edges.get(node_name, []):
                if cond is None or cond(result):
                    if parallel:
                        asyncio.create_task(_execute(next_node, result))
                    else:
                        result = await _execute(next_node, result)
            return result

        if self._entry_point is None:
            raise RuntimeError("Entry point not set")
        return asyncio.run(_execute(self._entry_point, input_text))


END = "END"
