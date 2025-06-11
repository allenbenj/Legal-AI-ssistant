"""Local fallback implementation of a minimal workflow graph."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Tuple


END = "END"


class BaseNode:
    """Lightweight base class used as a stand-in when ``langgraph`` is absent."""

    pass


# Helper to await values if needed
async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value) or isinstance(value, Awaitable):
        return await value
    return value


class StateGraph:
    """Simplified asynchronous graph structure for local workflows."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Callable[[Any], Any]] = {}
        self._edges: Dict[str, List[str]] = {}
        self._conditional_edges: Dict[
            str, List[Tuple[Callable[[Any], bool], str]]
        ] = {}
        self._parallel: Dict[str, Tuple[List[str], str]] = {}
        self._entry_point: str | None = None

    # ------------------------------------------------------------------
    # Graph construction helpers
    # ------------------------------------------------------------------
    def add_node(self, name: str, node: Callable[[Any], Any]) -> None:
        """Register a node callable under ``name``."""
        self._nodes[name] = node

    def add_edge(self, src: str, dest: str) -> None:
        """Connect ``src`` node to ``dest`` node."""
        self._edges.setdefault(src, []).append(dest)

    def add_conditional_edges(
        self,
        src: str,
        mapping: Iterable[Tuple[Callable[[Any], bool], str]],
    ) -> None:
        """Route to different nodes based on predicate results."""
        self._conditional_edges[src] = list(mapping)

    def add_parallel_nodes(self, src: str, nodes: List[str], merge: str) -> None:
        """Run ``nodes`` in parallel after ``src`` and send results to ``merge``."""
        self._parallel[src] = (nodes, merge)

    def set_entry_point(self, name: str) -> None:
        """Specify the starting node for execution."""
        self._entry_point = name

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def run(self, data: Any) -> Any:
        """Execute the graph synchronously starting with ``data``."""
        return asyncio.run(self._run(data))

    async def _run(self, data: Any) -> Any:
        if self._entry_point is None:
            raise RuntimeError("Entry point not set")

        node_name = self._entry_point
        state = data
        while node_name != END:
            node = self._nodes[node_name]
            state = await _maybe_await(node(state))

            # Handle parallel execution if configured
            if node_name in self._parallel:
                parallel_nodes, merge_node = self._parallel[node_name]
                results = await asyncio.gather(
                    *[
                        _maybe_await(self._nodes[n](state))
                        for n in parallel_nodes
                    ]
                )
                state = await _maybe_await(self._nodes[merge_node](results))
                node_name = self._next_node(merge_node, state)
                continue

            node_name = self._next_node(node_name, state)
        return state

    def _next_node(self, current: str, result: Any) -> str:
        """Determine the next node after ``current`` given ``result``."""
        if current in self._conditional_edges:
            for predicate, dest in self._conditional_edges[current]:
                if predicate(result):
                    return dest
            return END
        return self._edges.get(current, [END])[0]


__all__ = ["BaseNode", "StateGraph", "END"]
