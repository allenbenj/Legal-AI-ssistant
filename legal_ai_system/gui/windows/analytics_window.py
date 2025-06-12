from __future__ import annotations

"""Standalone PyQt6 window visualizing analytics data."""

import asyncio
from typing import Optional, Dict, Any

from PyQt6 import QtWidgets

from ..legal_ai_charts import (
    DocumentTypeChart,
    ProcessingStatsChart,
    WorkflowTrendChart,
)
from ...services.database_manager import DatabaseManager
from ...services.metrics_exporter import MetricsExporter
from ...services.realtime_graph_manager import RealTimeGraphManager


class AnalyticsWindow(QtWidgets.QWidget):
    """Window composing multiple analytics charts."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        metrics: Optional[MetricsExporter] = None,
        graph_manager: Optional[RealTimeGraphManager] = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.db = db_manager
        self.metrics = metrics
        self.graph_manager = graph_manager
        self.setWindowTitle("Analytics Dashboard")
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.doc_chart = DocumentTypeChart()
        self.proc_chart = ProcessingStatsChart()
        self.trend_chart = WorkflowTrendChart()
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.doc_chart)
        layout.addWidget(self.proc_chart)
        layout.addWidget(self.trend_chart)
        layout.addWidget(refresh_btn)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self._update_document_distribution()
        self._update_processing_stats()
        asyncio.create_task(self._update_workflow_trend())

    def _update_document_distribution(self) -> None:
        docs = self.db.get_documents()
        distribution: Dict[str, int] = {}
        for doc in docs:
            dtype = doc.get("file_type", "unknown")
            distribution[dtype] = distribution.get(dtype, 0) + 1
        self.doc_chart.update_data(distribution)

    def _update_processing_stats(self) -> None:
        if not self.metrics:
            return
        self.proc_chart.update_data(self.metrics.snapshot())

    async def _update_workflow_trend(self) -> None:
        if not self.graph_manager:
            return
        try:
            stats = await self.graph_manager.get_realtime_stats()
        except Exception:
            return
        value = stats.get("knowledge_graph_entities", 0)
        self.trend_chart.add_point(float(value))
