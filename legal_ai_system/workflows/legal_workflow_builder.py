from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List


class LegalWorkflowBuilder:
    """Simple sequential workflow runner."""

    def __init__(self) -> None:
        self._nodes: List[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = []

    def register_node(self, node: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]) -> None:
        """Register a node to run as part of the workflow."""
        self._nodes.append(node)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute nodes in order with the provided context."""
        for node in self._nodes:
            context = await node(context)
        return context

__all__ = ["LegalWorkflowBuilder"]
