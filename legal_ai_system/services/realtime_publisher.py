from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import psutil

try:
    from legal_ai_system.core.detailed_logging import (
        DetailedLogger,
        get_detailed_logger,
        LogCategory,
    )
except Exception:  # pragma: no cover - fallback
    import logging

    class LogCategory:  # type: ignore
        SYSTEM = "SYSTEM"

    class DetailedLogger(logging.Logger):
        def __init__(self, name: str, category: LogCategory = LogCategory.SYSTEM) -> None:
            super().__init__(name)
            self.category = category
            self.logger = self

    def get_detailed_logger(name: str, category: LogCategory) -> DetailedLogger:  # type: ignore
        return DetailedLogger(name, category)

from legal_ai_system.api.websocket_manager import ConnectionManager
from legal_ai_system.services.database_manager import DatabaseManager
from legal_ai_system.services.realtime_graph_manager import RealTimeGraphManager


class RealtimePublisher:
    """Publish system metrics periodically over WebSocket."""

    def __init__(
        self,
        manager: ConnectionManager,
        metrics_exporter: Optional[Any] = None,
        db_manager: Optional[DatabaseManager] = None,
        graph_manager: Optional[RealTimeGraphManager] = None,
    ) -> None:
        self.manager = manager
        self.metrics = metrics_exporter
        self.db_manager = db_manager
        self.graph_manager = graph_manager
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
            if self.metrics:
                snapshot = self.metrics.snapshot()
                await self.manager.broadcast("health_metrics", snapshot)
                await self.manager.broadcast("processing_stats", snapshot)
            if self.db_manager:
                await self.manager.broadcast(
                    "document_distribution", self._document_distribution()
                )
            if self.graph_manager:
                try:
                    graph_stats = await self.graph_manager.get_realtime_stats()
                    await self.manager.broadcast("workflow_trends", graph_stats)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.error("Failed to get graph stats", exc_info=exc)
            await asyncio.sleep(interval)

    def _collect_metrics(self) -> Dict[str, Any]:
        return {
            "type": "system_status",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage("/").percent,
        }

    def _document_distribution(self) -> Dict[str, Any]:
        if not self.db_manager:
            return {}
        docs = self.db_manager.get_documents()
        dist: Dict[str, int] = {}
        for doc in docs:
            dtype = doc.get("file_type", "unknown")
            dist[dtype] = dist.get(dtype, 0) + 1
        return {"type": "document_distribution", "data": dist}
