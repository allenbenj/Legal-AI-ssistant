"""Minimal fallback implementation of a workflow graph engine."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple


END = "END"


class BaseNode:
    """Placeholder base class used when the real package is unavailable."""

    pass


async def _maybe_await(value: Awaitable | Any) -> Any:
    """Await ``value`` if needed and return the result."""

    if inspect.isawaitable(value):
        return await value
    return value


class StateGraph:
    """Very small workflow graph for environments without LangGraph."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Callable[[Any], Any]] = {}
        self._edges: Dict[str, List[str]] = {}
        self._conditional_edges: Dict[
            str, List[Tuple[Callable[[Any], bool], str]]
        ] = {}
        self._parallel: Dict[str, Tuple[List[str], str]] = {}
        self._entry_point: Optional[str] = None

    def add_node(self, name: str, node: Callable[[Any], Any]) -> None:
        """Register ``node`` under ``name``."""

        self._nodes[name] = node

    def add_edge(self, src: str, dest: str) -> None:
        """Connect ``src`` node to ``dest`` node."""

        self._edges.setdefault(src, []).append(dest)

    def add_conditional_edges(
        self, src: str, mapping: Iterable[Tuple[Callable[[Any], bool], str]]
    ) -> None:
        """Route to different nodes based on predicate results."""

        self._conditional_edges[src] = list(mapping)

    def add_parallel_nodes(self, src: str, nodes: List[str], merge: str) -> None:
        """Run ``nodes`` in parallel after ``src`` and merge via ``merge``."""

        self._parallel[src] = (nodes, merge)

    def set_entry_point(self, name: str) -> None:
        self._entry_point = name

    def run(self, state: Any) -> Any:
        """Execute the graph synchronously starting with ``state``."""

        return asyncio.run(self._run_async(state))

    async def _run_async(self, state: Any) -> Any:
        if self._entry_point is None:
            raise RuntimeError("Entry point not set")

        node_name = self._entry_point
        while node_name != END:
            func = self._nodes[node_name]
            state = await _maybe_await(func(state))

            # Handle parallel execution if configured
            if node_name in self._parallel:
                parallel_nodes, merge_node = self._parallel[node_name]
                results = await asyncio.gather(
                    *[_maybe_await(self._nodes[n](state)) for n in parallel_nodes]
                )
                state = await _maybe_await(self._nodes[merge_node](results))
                node_name = self._next_node(merge_node, state)
                continue

            node_name = self._next_node(node_name, state)

        return state

    def _next_node(self, current: str, result: Any) -> str:
        """Return the next node name after ``current`` based on ``result``."""

        if current in self._conditional_edges:
            for predicate, dest in self._conditional_edges[current]:
                if predicate(result):
                    return dest
            return END
        return self._edges.get(current, [END])[0]


__all__ = ["BaseNode", "StateGraph", "END"]

