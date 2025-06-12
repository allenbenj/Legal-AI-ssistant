from __future__ import annotations

"""Dashboard helper for :mod:`legal_ai_system.gui`."""

from PyQt6.QtCore import QDateTime
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from ..legal_ai_charts import BarChartWidget, PieChartWidget, ChartData
from legal_ai_widgets import FlipCard, TagCloud, TimelineWidget


class DashboardSection:
    """Encapsulates dashboard creation and updates."""

    def __init__(self, main_window: QWidget) -> None:
        self.main_window = main_window
        self.widget: QWidget | None = None
        self.doc_count_card: FlipCard | None = None
        self.success_rate_card: FlipCard | None = None
        self.active_users_card: FlipCard | None = None
        self.timeline: TimelineWidget | None = None
        self.tag_cloud: TagCloud | None = None
        self.mini_pie: PieChartWidget | None = None
        self.mini_bar: BarChartWidget | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def create_widget(self) -> QWidget:
        dashboard = QWidget()
        layout = QGridLayout(dashboard)

        stats_frame = QFrame()
        stats_frame.setFrameStyle(QFrame.Shape.Box)
        stats_layout = QHBoxLayout(stats_frame)

        self.doc_count_card = FlipCard("Total Documents\n0", "Click for details")
        stats_layout.addWidget(self.doc_count_card)

        self.success_rate_card = FlipCard(
            "Success Rate\n0%", "Processing accuracy"
        )
        stats_layout.addWidget(self.success_rate_card)

        self.active_users_card = FlipCard("Active Users\n0", "Currently online")
        stats_layout.addWidget(self.active_users_card)

        layout.addWidget(stats_frame, 0, 0, 1, 2)

        self.timeline = TimelineWidget()
        timeline_frame = QGroupBox("Recent Activity")
        timeline_layout = QVBoxLayout(timeline_frame)
        timeline_layout.addWidget(self.timeline)
        layout.addWidget(timeline_frame, 1, 0)

        self.tag_cloud = TagCloud()
        tag_frame = QGroupBox("Popular Tags")
        tag_layout = QVBoxLayout(tag_frame)
        tag_layout.addWidget(self.tag_cloud)
        layout.addWidget(tag_frame, 1, 1)

        self.mini_pie = PieChartWidget()
        self.mini_pie.setMaximumHeight(300)
        layout.addWidget(self.mini_pie, 2, 0)

        self.mini_bar = BarChartWidget()
        self.mini_bar.setMaximumHeight(300)
        layout.addWidget(self.mini_bar, 2, 1)

        self.widget = dashboard
        return dashboard

    # ------------------------------------------------------------------
    # Update logic
    # ------------------------------------------------------------------
    def update(self) -> None:
        if not self.widget:
            return

        documents = self.main_window.document_section.documents
        doc_count = len(documents)
        if self.doc_count_card:
            self.doc_count_card.front_content = f"Total Documents\n{doc_count}"

        if self.timeline:
            now = QDateTime.currentDateTime()
            for i, (_, doc) in enumerate(list(documents.items())[:5]):
                self.timeline.addEvent(
                    now.addSecs(-i * 3600),
                    doc.filename,
                    f"Status: {doc.status}",
                    "success" if doc.status == "completed" else "info",
                )

        if self.tag_cloud:
            tags = [
                {"text": "Contract", "weight": 2.0},
                {"text": "Legal", "weight": 1.5},
                {"text": "Compliance", "weight": 1.0},
                {"text": "Patent", "weight": 0.8},
            ]
            self.tag_cloud.setTags(tags)

        if self.mini_pie:
            pie_data = [
                ChartData("Completed", 75),
                ChartData("Processing", 15),
                ChartData("Pending", 10),
            ]
            self.mini_pie.setData(pie_data)

        if self.mini_bar:
            bar_data = [
                ChartData("Mon", 12),
                ChartData("Tue", 19),
                ChartData("Wed", 15),
                ChartData("Thu", 25),
                ChartData("Fri", 22),
            ]
            self.mini_bar.setData(bar_data)
