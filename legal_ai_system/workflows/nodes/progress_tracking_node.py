from __future__ import annotations

from typing import Any, Dict

try:  # pragma: no cover - avoid mandatory fastapi dependency in tests
    from ...api.websocket_manager import ConnectionManager
except Exception:
    from typing import Protocol

    class ConnectionManager(Protocol):  # type: ignore[misc]
        async def broadcast(self, topic: str, message: Dict[str, Any]) -> None:
            ...


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

