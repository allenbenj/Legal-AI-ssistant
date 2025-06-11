"""Local fallback implementation of a minimal workflow graph."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Tuple


END = "END"


"""Minimal local stub for the optional :mod:`langgraph` package."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple


class BaseNode:
    pass


    def __init__(self) -> None:
        self._nodes: Dict[str, Callable[[Any], Any]] = {}
        self._edges: Dict[str, List[str]] = {}

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
