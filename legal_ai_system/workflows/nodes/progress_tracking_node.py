from __future__ import annotations

from typing import Any, Dict

from ...api.websocket_manager import ConnectionManager


class ProgressTrackingNode:
    """Broadcast workflow progress updates over WebSocket."""

    def __init__(self, manager: ConnectionManager, topic: str) -> None:
        self.manager = manager
        self.topic = topic

    async def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send the progress update to all subscribed clients and return the data."""

        message_type = "progress_update" if "document_id" in data else "progress"
        await self.manager.broadcast(self.topic, {"type": message_type, **data})
        return data

