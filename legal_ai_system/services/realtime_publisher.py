from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import psutil

try:
    from legal_ai_system.core.detailed_logging import get_detailed_logger, LogCategory
except Exception:  # pragma: no cover - fallback
    import logging
    class LogCategory:  # type: ignore
        SYSTEM = "SYSTEM"
    def get_detailed_logger(name: str, category: LogCategory):  # type: ignore
        return logging.getLogger(name)

from legal_ai_system.api.websocket_manager import ConnectionManager


class RealtimePublisher:
    """Publish system metrics periodically over WebSocket."""

    def __init__(self, manager: ConnectionManager) -> None:
        self.manager = manager
        self.logger = get_detailed_logger("RealtimePublisher", LogCategory.SYSTEM)
        self._task: Optional[asyncio.Task] = None

    def start_system_monitoring(self, interval: float = 1.0) -> None:
        """Start background task broadcasting system metrics."""
        if self._task:
            return
        self._task = asyncio.create_task(self._monitor_loop(interval))
        self.logger.info("Started system monitoring task")

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None
            self.logger.info("Stopped system monitoring task")

    async def _monitor_loop(self, interval: float) -> None:
        while True:
            metrics = self._collect_metrics()
            await self.manager.broadcast("system_status", metrics)
            await asyncio.sleep(interval)

    def _collect_metrics(self) -> Dict[str, Any]:
        return {
            "type": "system_status",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage("/").percent,
        }
