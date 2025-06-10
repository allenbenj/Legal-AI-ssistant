from __future__ import annotations

from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    from langgraph.graph import BaseNode
except Exception:  # pragma: no cover - fallback stub
    class BaseNode:
        pass

from ...api.websocket_manager import ConnectionManager


class ProgressTrackingNode(BaseNode):
    """Broadcast workflow progress updates via :class:`ConnectionManager`."""

    def __init__(self, manager: ConnectionManager, topic: str = "workflow_progress") -> None:
        self.manager = manager
        self.topic = topic

    async def __call__(self, progress: Dict[str, Any]) -> Dict[str, Any]:
        await self.manager.broadcast(self.topic, {"type": "progress_update", **progress})
        return progress


__all__ = ["ProgressTrackingNode"]
