from __future__ import annotations

"""Reusable PyQt chart widgets for analytics windows."""

from typing import Dict, List

from PyQt6 import QtCore, QtGui, QtCharts, QtWidgets


class DocumentTypeChart(QtCharts.QChartView):
    """Pie chart showing distribution of processed document types."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        chart = QtCharts.QChart()
        super().__init__(chart, parent)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.series = QtCharts.QPieSeries()
        chart.addSeries(self.series)
        chart.setTitle("Document Type Distribution")

    def update_data(self, distribution: Dict[str, int]) -> None:
        self.series.clear()
        for doc_type, count in distribution.items():
            self.series.append(doc_type, count)


class ProcessingStatsChart(QtCharts.QChartView):
    """Bar chart visualizing processing statistics from :class:`MetricsExporter`."""

    METRIC_LABELS = {
        "kg_queries_total": "KG Queries",
        "kg_query_cache_hits": "KG Cache Hits",
        "vector_add_seconds_sum": "Vector Add (s)",
        "vector_search_seconds_sum": "Vector Search (s)",
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        chart = QtCharts.QChart()
        super().__init__(chart, parent)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.series = QtCharts.QBarSeries()
        chart.addSeries(self.series)
        axis_x = QtCharts.QBarCategoryAxis()
        axis_y = QtCharts.QValueAxis()
        chart.addAxis(axis_x, QtCore.Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.series.attachAxis(axis_x)
        self.series.attachAxis(axis_y)
        chart.setTitle("Processing Statistics")
        self.axis_x = axis_x
        self.axis_y = axis_y

    def update_data(self, metrics: Dict[str, float]) -> None:
        self.series.clear()
        set0 = QtCharts.QBarSet("Metrics")
        categories: List[str] = []
        for key, label in self.METRIC_LABELS.items():
            value = float(metrics.get(key, 0))
            set0 << value
            categories.append(label)
        self.series.append(set0)
        self.axis_x.clear()
        self.axis_x.append(categories)
        max_val = max([float(metrics.get(k, 0)) for k in self.METRIC_LABELS], default=1)
        self.axis_y.setRange(0, max_val * 1.2)


class WorkflowTrendChart(QtCharts.QChartView):
    """Line chart plotting workflow trends over time from :class:`RealTimeGraphManager`."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        chart = QtCharts.QChart()
        super().__init__(chart, parent)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.series = QtCharts.QLineSeries()
        chart.addSeries(self.series)
        axis_x = QtCharts.QValueAxis()
        axis_y = QtCharts.QValueAxis()
        chart.addAxis(axis_x, QtCore.Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.series.attachAxis(axis_x)
        self.series.attachAxis(axis_y)
        chart.setTitle("Workflow Trends")
        self.index = 0
        self.axis_y = axis_y

    def add_point(self, value: float) -> None:
        self.series.append(float(self.index), float(value))
        self.index += 1
        ymax = max(point.y() for point in self.series.pointsVector())
        self.axis_y.setRange(0, ymax * 1.2 if ymax > 0 else 1)
