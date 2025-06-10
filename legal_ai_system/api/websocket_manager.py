from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Set

from fastapi import WebSocket, WebSocketDisconnect

try:
    from legal_ai_system.core.detailed_logging import (
        DetailedLogger,
        get_detailed_logger,
        LogCategory,
    )
except Exception:  # pragma: no cover - fall back to std logging
    import logging

    class LogCategory:  # type: ignore
        API = "API"

    class DetailedLogger(logging.Logger):
        def __init__(self, name: str, category: LogCategory = LogCategory.API) -> None:
            super().__init__(name)
            self.category = category
            self.logger = self

    def get_detailed_logger(name: str, category: LogCategory) -> DetailedLogger:  # type: ignore
        return DetailedLogger(name, category)


class ConnectionManager:
    """Manage WebSocket connections and topic subscriptions."""

    def __init__(self) -> None:
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.topic_subscribers: Dict[str, Set[str]] = defaultdict(set)
        self.logger = get_detailed_logger("ConnectionManager", LogCategory.API)

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.logger.info("Client connected", parameters={"client_id": client_id})

    def disconnect(self, client_id: str) -> None:
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        topics = self.subscriptions.pop(client_id, set())
        for topic in topics:
            self.topic_subscribers[topic].discard(client_id)
            if not self.topic_subscribers[topic]:
                del self.topic_subscribers[topic]
        self.logger.info("Client disconnected", parameters={"client_id": client_id})

    async def send_personal_message(self, message: Dict[str, Any], client_id: str) -> None:
        websocket = self.active_connections.get(client_id)
        if not websocket:
            return
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except WebSocketDisconnect:
            self.disconnect(client_id)
        except Exception:
            self.disconnect(client_id)

    async def broadcast(self, topic: str, message: Dict[str, Any]) -> None:
        if topic not in self.topic_subscribers:
            return
        for client_id in list(self.topic_subscribers[topic]):
            await self.send_personal_message(message, client_id)

    # Backwards compatibility aliases
    async def broadcast_to_topic(self, message: Dict[str, Any], topic: str) -> None:
        await self.broadcast(topic, message)

    async def subscribe(self, client_id: str, topic: str) -> None:
        self.subscriptions[client_id].add(topic)
        self.topic_subscribers[topic].add(client_id)
        await self.send_personal_message(
            {"type": "subscription_ack", "topic": topic}, client_id
        )

    async def subscribe_to_topic(self, user_id: str, topic: str) -> None:
        await self.subscribe(user_id, topic)

    async def unsubscribe(self, client_id: str, topic: str) -> None:
        self.subscriptions[client_id].discard(topic)
        if topic in self.topic_subscribers:
            self.topic_subscribers[topic].discard(client_id)
            if not self.topic_subscribers[topic]:
                del self.topic_subscribers[topic]
        await self.send_personal_message(
            {"type": "subscription_ack", "topic": topic, "status": "unsubscribed"},
            client_id,
        )

    async def unsubscribe_from_topic(self, user_id: str, topic: str) -> None:
        await self.unsubscribe(user_id, topic)
