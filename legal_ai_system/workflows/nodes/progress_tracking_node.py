from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

try:
    from ...api.websocket_manager import ConnectionManager
except Exception:  # pragma: no cover - tests may stub this
    ConnectionManager = Any  # type: ignore


@dataclass
class ProgressTrackingNode:
    """Node that broadcasts progress updates via WebSocket."""

    manager: ConnectionManager
    topic: str = "workflow_progress"

    async def __call__(self, update: Dict[str, Any]) -> Dict[str, Any]:
        await self.manager.broadcast(self.topic, {"type": "progress", **update})
        return update
