


"""Minimal local stub for the optional :mod:`langgraph` package."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple


class BaseNode:
    """Lightweight interface for callable workflow nodes."""

    pass



class StateGraph:
    """Simple asynchronous graph executor used for local development."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Callable[[Any], Any]] = {}
        self._edges: Dict[str, List[str]] = {}
        self._cond_edges: Dict[str, List[Tuple[Callable[[Any], bool], str]]] = {}
        self._parallel: Dict[str, Tuple[List[str], str]] = {}
        self._entry_point: Optional[str] = None

    # ------------------------------------------------------------------
    # Node registration utilities
    # ------------------------------------------------------------------
    def add_node(self, name: str, node: Callable[[Any], Any]) -> None:
        """Register a callable node by name."""

        self._nodes[name] = node

    def set_entry_point(self, name: str) -> None:
        """Set the starting node for graph execution."""

        self._entry_point = name

    def add_edge(self, start: str, end: str) -> None:
        """Add a linear edge from ``start`` to ``end``."""

        self._edges.setdefault(start, []).append(end)

    def add_conditional_edges(
        self,
        start: str,
        mapping: Iterable[Tuple[Callable[[Any], bool], str]],
    ) -> None:
        """Add conditional transitions executed in order."""

        self._cond_edges[start] = list(mapping)

    def add_parallel_nodes(self, start: str, nodes: List[str], merge: str) -> None:
        """Execute ``nodes`` concurrently and merge the results."""

        self._parallel[start] = (nodes, merge)

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    async def _call_node(self, name: str, data: Any) -> Any:
        func = self._nodes[name]
        result = func(data)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _run_async(self, data: Any) -> Any:
        if self._entry_point is None:
            raise RuntimeError("Entry point not set")

        node = self._entry_point
        result = data

        while node != END:
            result = await self._call_node(node, result)

            # Handle fan-out/fan-in style parallel execution
            if node in self._parallel:
                branches, merge = self._parallel[node]
                branch_results = await asyncio.gather(
                    *[self._call_node(n, result) for n in branches]
                )
                result = await self._call_node(merge, branch_results)
                node = merge

            if node in self._cond_edges:
                dest = END
                for cond, target in self._cond_edges[node]:
                    try:
                        if cond(result):
                            dest = target
                            break
                    except Exception:
                        continue
                node = dest
            elif node in self._edges:
                node = self._edges[node][0]
            else:
                node = END
        return result

    # Public API --------------------------------------------------------
    def run(self, data: Any) -> Any:
        """Execute the graph synchronously."""

        return asyncio.run(self._run_async(data))

    async def arun(self, data: Any) -> Any:
        """Asynchronously execute the graph."""

        return await self._run_async(data)


END = "END"
