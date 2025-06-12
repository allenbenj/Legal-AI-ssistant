from __future__ import annotations

"""Reusable PyQt6 chart widgets used across the GUI."""

from typing import Iterable, Mapping, Tuple

from PyQt6 import QtCore, QtGui, QtCharts, QtWidgets


class PieChartWidget(QtCharts.QChartView):
    """Simple pie chart widget with click notifications."""

    slice_clicked = QtCore.pyqtSignal(str)

    def __init__(self, data: Mapping[str, float] | Iterable[Tuple[str, float]] | None = None,
                 parent: QtWidgets.QWidget | None = None) -> None:
        chart = QtCharts.QChart()
        super().__init__(chart, parent)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.series = QtCharts.QPieSeries()
        chart.addSeries(self.series)
        chart.legend().setVisible(True)
        chart.setAnimationOptions(QtCharts.QChart.AnimationOption.SeriesAnimations)
        if data:
            self.set_data(data)

    def set_data(self, data: Mapping[str, float] | Iterable[Tuple[str, float]]) -> None:
        """Populate the chart with slices."""
        self.series.clear()
        items = data.items() if isinstance(data, Mapping) else data
        for label, value in items:
            slice_ = QtCharts.QPieSlice(label, value)
            slice_.clicked.connect(lambda _=False, l=label: self.slice_clicked.emit(l))
            self.series.append(slice_)


class BarChartWidget(QtCharts.QChartView):
    """Basic bar chart widget with click notifications."""

    bar_clicked = QtCore.pyqtSignal(str)

    def __init__(self, categories: Iterable[str] | None = None,
                 data: Mapping[str, float] | Iterable[Tuple[str, float]] | None = None,
                 parent: QtWidgets.QWidget | None = None) -> None:
        chart = QtCharts.QChart()
        super().__init__(chart, parent)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.series = QtCharts.QBarSeries()
        chart.addSeries(self.series)
        self.axis = QtCharts.QBarCategoryAxis()
        chart.createDefaultAxes()
        chart.setAxisX(self.axis, self.series)
        chart.legend().setVisible(True)
        chart.setAnimationOptions(QtCharts.QChart.AnimationOption.SeriesAnimations)
        if categories:
            self.axis.append(list(categories))
        if data:
            self.set_data(data)

    def set_categories(self, categories: Iterable[str]) -> None:
        self.axis.clear()
        self.axis.append(list(categories))

    def set_data(self, data: Mapping[str, float] | Iterable[Tuple[str, float]]) -> None:
        """Add bars to the chart."""
        self.series.clear()
        bar_set = QtCharts.QBarSet("Values")
        items = data.items() if isinstance(data, Mapping) else data
        for label, value in items:
            bar_set << float(value)
        bar_set.clicked.connect(lambda index: self._emit_click(index))
        self.series.append(bar_set)
        self.set_categories([label for label, _ in items])

    def _emit_click(self, index: int) -> None:
        labels = [self.axis.at(i) for i in range(self.axis.count())]
        if 0 <= index < len(labels):
            self.bar_clicked.emit(labels[index])


class AnalyticsDashboardWidget(QtWidgets.QWidget):
    """Dashboard widget aggregating different charts."""

    chart_item_clicked = QtCore.pyqtSignal(str, str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.pie_chart = PieChartWidget()
        self.bar_chart = BarChartWidget()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.pie_chart)
        layout.addWidget(self.bar_chart)
        self.pie_chart.slice_clicked.connect(
            lambda label: self.chart_item_clicked.emit("pie", label)
        )
        self.bar_chart.bar_clicked.connect(
            lambda label: self.chart_item_clicked.emit("bar", label)
        )

    def set_pie_data(self, data: Mapping[str, float] | Iterable[Tuple[str, float]]) -> None:
        self.pie_chart.set_data(data)

    def set_bar_data(self, data: Mapping[str, float] | Iterable[Tuple[str, float]]) -> None:
        self.bar_chart.set_data(data)

