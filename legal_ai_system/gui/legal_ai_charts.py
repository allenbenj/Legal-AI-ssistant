from __future__ import annotations

"""Data visualization widgets for Legal AI System."""

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


# ==================== DATA STRUCTURES ====================
@dataclass
class ChartData:
    """Data structure for chart values"""

    label: str
    value: float
    color: Optional[QColor] = None
    metadata: Dict[str, Any] | None = None


# ==================== CUSTOM CHART WIDGETS ====================


class PieChartWidget(QWidget):
    """Interactive pie chart for document type distribution"""

    segmentClicked = pyqtSignal(str, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.data: List[ChartData] = []
        self.hover_segment: ChartData | None = None
        self.animation_progress = 0.0
        self.setMouseTracking(True)
        self.setMinimumSize(300, 300)

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.updateAnimation)

        self.default_colors = [
            QColor("#dc143c"),
            QColor("#2196f3"),
            QColor("#4caf50"),
            QColor("#ff9800"),
            QColor("#9c27b0"),
            QColor("#00bcd4"),
            QColor("#ffeb3b"),
            QColor("#795548"),
        ]

    def setData(self, data: List[ChartData]) -> None:
        """Set chart data and start animation"""
        self.data = data
        self.animation_progress = 0.0

        for i, item in enumerate(self.data):
            if item.color is None:
                item.color = self.default_colors[i % len(self.default_colors)]

        total = sum(item.value for item in self.data)
        for item in self.data:
            item.percentage = (item.value / total * 100) if total > 0 else 0

        self.animation_timer.start(20)

    def updateAnimation(self) -> None:
        self.animation_progress += 0.05
        if self.animation_progress >= 1.0:
            self.animation_progress = 1.0
            self.animation_timer.stop()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        center = QPoint(self.width() // 2, self.height() // 2)
        mouse_pos = event.pos()

        dx = mouse_pos.x() - center.x()
        dy = mouse_pos.y() - center.y()
        distance = math.sqrt(dx * dx + dy * dy)

        radius = min(self.width(), self.height()) // 2 - 40
        if radius * 0.3 <= distance <= radius:
            angle = math.degrees(math.atan2(dy, dx))
            if angle < 0:
                angle += 360

            current_angle = 0.0
            self.hover_segment = None
            for item in self.data:
                segment_angle = item.percentage * 3.6
                if current_angle <= angle <= current_angle + segment_angle:
                    self.hover_segment = item
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                    self.setToolTip(
                        f"{item.label}: {item.value} ({item.percentage:.1f}%)"
                    )
                    break
                current_angle += segment_angle
        else:
            self.hover_segment = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.setToolTip("")

        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.hover_segment and event.button() == Qt.MouseButton.LeftButton:
            self.segmentClicked.emit(self.hover_segment.label, self.hover_segment.value)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        center = rect.center()
        radius = min(rect.width(), rect.height()) // 2 - 40

        start_angle = 0
        for item in self.data:
            span_angle = int(item.percentage * 3.6 * 16 * self.animation_progress)

            offset = 10 if item == self.hover_segment else 0
            if offset:
                angle_rad = math.radians(start_angle / 16 + span_angle / 32)
                offset_x = offset * math.cos(angle_rad)
                offset_y = offset * math.sin(angle_rad)
            else:
                offset_x = offset_y = 0

            segment_rect = QRect(
                center.x() - radius + int(offset_x),
                center.y() - radius + int(offset_y),
                radius * 2,
                radius * 2,
            )

            painter.setBrush(item.color)
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawPie(segment_rect, start_angle, span_angle)

            if self.animation_progress >= 1.0 and item.percentage > 5:
                label_angle = start_angle / 16 + span_angle / 32
                label_radius = radius * 0.7
                label_x = center.x() + label_radius * math.cos(
                    math.radians(label_angle)
                )
                label_y = center.y() + label_radius * math.sin(
                    math.radians(label_angle)
                )

                painter.setPen(Qt.GlobalColor.white)
                painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                painter.drawText(
                    QRect(int(label_x - 30), int(label_y - 10), 60, 20),
                    Qt.AlignmentFlag.AlignCenter,
                    f"{item.percentage:.0f}%",
                )

            start_angle += span_angle

        painter.setBrush(QColor("#1a1a1a"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, int(radius * 0.3), int(radius * 0.3))

        painter.setPen(Qt.GlobalColor.white)
        painter.setFont(QFont("Arial", 12))
        painter.drawText(
            QRect(center.x() - 50, center.y() - 10, 100, 20),
            Qt.AlignmentFlag.AlignCenter,
            "Documents",
        )


class BarChartWidget(QWidget):
    """Animated bar chart for processing statistics"""

    barClicked = pyqtSignal(str, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.data: List[ChartData] = []
        self.hover_bar: ChartData | None = None
        self.animation_values: dict[str, float] = {}
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.updateAnimation)
        self.animation_timer.setInterval(20)

    def setData(self, data: List[ChartData]) -> None:
        """Set chart data and animate"""
        self.data = data
        for item in self.data:
            if item.label not in self.animation_values:
                self.animation_values[item.label] = 0.0
        self.animation_timer.start()

    def updateAnimation(self) -> None:
        all_complete = True
        for item in self.data:
            current = self.animation_values.get(item.label, 0.0)
            target = item.value

            if current < target:
                self.animation_values[item.label] = min(current + target * 0.05, target)
                all_complete = False
            elif current > target:
                self.animation_values[item.label] = max(
                    current - current * 0.05, target
                )
                all_complete = False

        if all_complete:
            self.animation_timer.stop()

        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self.data:
            return

        bar_width = (self.width() - 100) // len(self.data)
        x_offset = 50

        self.hover_bar = None
        for i, item in enumerate(self.data):
            bar_x = x_offset + i * bar_width
            bar_rect = QRect(bar_x, 0, bar_width - 10, self.height() - 60)

            if bar_rect.contains(event.pos()):
                self.hover_bar = item
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                self.setToolTip(f"{item.label}: {item.value}")
                break
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.setToolTip("")

        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.hover_bar and event.button() == Qt.MouseButton.LeftButton:
            self.barClicked.emit(self.hover_bar.label, self.hover_bar.value)

    def paintEvent(self, event: QPaintEvent) -> None:
        if not self.data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        margin = 50
        chart_height = self.height() - margin * 2
        chart_width = self.width() - margin * 2
        bar_width = chart_width // len(self.data)

        max_value = max(self.animation_values.values()) if self.animation_values else 1

        painter.setPen(QPen(QColor("#444"), 2))
        painter.drawLine(margin, margin, margin, self.height() - margin)
        painter.drawLine(
            margin,
            self.height() - margin,
            self.width() - margin,
            self.height() - margin,
        )

        for i, item in enumerate(self.data):
            value = self.animation_values.get(item.label, 0.0)
            bar_height = int((value / max_value) * chart_height * 0.8)

            x = margin + i * bar_width + bar_width // 4
            y = self.height() - margin - bar_height
            width = bar_width // 2

            color = item.color if item.color else QColor("#dc143c")
            if item == self.hover_bar:
                color = color.lighter(120)

            bar_rect = QRect(x, y, width, bar_height)

            gradient = QLinearGradient(0, y, 0, y + bar_height)
            gradient.setColorAt(0, color.lighter(110))
            gradient.setColorAt(1, color)
            painter.fillRect(bar_rect, gradient)

            painter.setPen(Qt.GlobalColor.white)
            painter.setFont(QFont("Arial", 10))
            painter.drawText(
                QRect(x, y - 20, width, 20),
                Qt.AlignmentFlag.AlignCenter,
                f"{value:.0f}",
            )

            painter.setFont(QFont("Arial", 9))
            label_rect = QRect(x - 10, self.height() - margin + 5, width + 20, 30)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, item.label)

        painter.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        painter.drawText(
            QRect(0, 10, self.width(), 30),
            Qt.AlignmentFlag.AlignCenter,
            "Processing Statistics",
        )


class LineChartWidget(QWidget):
    """Time series line chart for trends"""

    pointClicked = pyqtSignal(datetime, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.series: Dict[str, List[Tuple[datetime, float]]] = {}
        self.series_colors: Dict[str, QColor] = {}
        self.hover_point: tuple[str, int, datetime, float] | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(500, 300)

        self.color_palette = [
            QColor("#dc143c"),
            QColor("#2196f3"),
            QColor("#4caf50"),
            QColor("#ff9800"),
        ]

    def addSeries(
        self, name: str, data: List[Tuple[datetime, float]], color: QColor | None = None
    ) -> None:
        """Add a data series to the chart"""
        self.series[name] = sorted(data, key=lambda x: x[0])
        if color is None:
            color_index = len(self.series_colors) % len(self.color_palette)
            color = self.color_palette[color_index]
        self.series_colors[name] = color
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self.series:
            return

        margin = 50
        chart_rect = QRect(
            margin, margin, self.width() - 2 * margin, self.height() - 2 * margin
        )

        if chart_rect.contains(event.pos()):
            min_distance = float("inf")
            self.hover_point = None
            for series_name, points in self.series.items():
                for i, (timestamp, value) in enumerate(points):
                    x, y = self._getPointPosition(timestamp, value, chart_rect)
                    distance = math.sqrt(
                        (x - event.pos().x()) ** 2 + (y - event.pos().y()) ** 2
                    )
                    if distance < min_distance and distance < 10:
                        min_distance = distance
                        self.hover_point = (series_name, i, timestamp, value)
            if self.hover_point:
                self.setCursor(Qt.CursorShape.CrossCursor)
                _, _, timestamp, value = self.hover_point
                self.setToolTip(f"{timestamp.strftime('%Y-%m-%d %H:%M')}: {value:.2f}")
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.setToolTip("")
        else:
            self.hover_point = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def _getPointPosition(
        self, timestamp: datetime, value: float, chart_rect: QRect
    ) -> Tuple[int, int]:
        all_times: list[datetime] = []
        all_values: list[float] = []
        for points in self.series.values():
            all_times.extend([t for t, _ in points])
            all_values.extend([v for _, v in points])
        if not all_times:
            return 0, 0
        min_time = min(all_times)
        max_time = max(all_times)
        min_value = min(all_values)
        max_value = max(all_values)

        value_range = max_value - min_value
        if value_range == 0:
            value_range = 1
        min_value -= value_range * 0.1
        max_value += value_range * 0.1

        time_range = (max_time - min_time).total_seconds()
        if time_range == 0:
            time_range = 1

        x = (
            chart_rect.left()
            + (timestamp - min_time).total_seconds() / time_range * chart_rect.width()
        )
        y = (
            chart_rect.bottom()
            - (value - min_value) / (max_value - min_value) * chart_rect.height()
        )
        return int(x), int(y)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        margin = 50
        chart_rect = QRect(
            margin, margin, self.width() - 2 * margin, self.height() - 2 * margin
        )
        painter.fillRect(chart_rect, QColor("#2d2d2d"))

        painter.setPen(QPen(QColor("#444"), 1, Qt.PenStyle.DashLine))
        for i in range(5):
            y = chart_rect.top() + i * chart_rect.height() // 4
            painter.drawLine(chart_rect.left(), y, chart_rect.right(), y)
            x = chart_rect.left() + i * chart_rect.width() // 4
            painter.drawLine(x, chart_rect.top(), x, chart_rect.bottom())

        for series_name, points in self.series.items():
            if not points:
                continue
            color = self.series_colors.get(series_name, QColor("#dc143c"))
            painter.setPen(QPen(color, 2))
            path = QPainterPath()
            for i, (timestamp, value) in enumerate(points):
                x, y = self._getPointPosition(timestamp, value, chart_rect)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.drawPath(path)

            painter.setBrush(color)
            for i, (timestamp, value) in enumerate(points):
                x, y = self._getPointPosition(timestamp, value, chart_rect)
                if (
                    self.hover_point
                    and self.hover_point[0] == series_name
                    and self.hover_point[1] == i
                ):
                    painter.setPen(QPen(Qt.GlobalColor.white, 2))
                    painter.drawEllipse(QPoint(x, y), 6, 6)
                else:
                    painter.setPen(QPen(color, 1))
                    painter.drawEllipse(QPoint(x, y), 4, 4)

        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawLine(chart_rect.bottomLeft(), chart_rect.topLeft())
        painter.drawLine(chart_rect.bottomLeft(), chart_rect.bottomRight())

        legend_y = 10
        for series_name, color in self.series_colors.items():
            painter.fillRect(self.width() - 150, legend_y, 20, 15, color)
            painter.setPen(Qt.GlobalColor.white)
            painter.setFont(QFont("Arial", 10))
            painter.drawText(self.width() - 120, legend_y + 12, series_name)
            legend_y += 20


class HeatMapWidget(QWidget):
    """Heat map for correlation or activity visualization"""

    cellClicked = pyqtSignal(int, int, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.data: List[List[float]] = []
        self.row_labels: List[str] = []
        self.col_labels: List[str] = []
        self.hover_cell: tuple[int, int] | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(400, 400)

    def setData(
        self, data: List[List[float]], row_labels: List[str], col_labels: List[str]
    ) -> None:
        """Set heat map data"""
        self.data = data
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self.data:
            return
        margin = 80
        cell_width = (self.width() - 2 * margin) // len(self.col_labels)
        cell_height = (self.height() - 2 * margin) // len(self.row_labels)
        x = event.pos().x() - margin
        y = event.pos().y() - margin
        if x >= 0 and y >= 0:
            col = x // cell_width
            row = y // cell_height
            if 0 <= row < len(self.row_labels) and 0 <= col < len(self.col_labels):
                self.hover_cell = (row, col)
                value = self.data[row][col]
                self.setToolTip(
                    f"{self.row_labels[row]} × {self.col_labels[col]}: {value:.2f}"
                )
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                self.update()
                return
        self.hover_cell = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setToolTip("")
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        if not self.data:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        margin = 80
        cell_width = (self.width() - 2 * margin) // len(self.col_labels)
        cell_height = (self.height() - 2 * margin) // len(self.row_labels)
        flat_data = [val for row in self.data for val in row]
        min_val = min(flat_data) if flat_data else 0
        max_val = max(flat_data) if flat_data else 1
        value_range = max_val - min_val if max_val != min_val else 1
        for i, row in enumerate(self.data):
            for j, value in enumerate(row):
                x = margin + j * cell_width
                y = margin + i * cell_height
                intensity = (value - min_val) / value_range
                color = QColor()
                color.setHsv(0, int(255 * intensity), int(200 + 55 * (1 - intensity)))
                if self.hover_cell == (i, j):
                    painter.fillRect(
                        x - 2,
                        y - 2,
                        cell_width + 4,
                        cell_height + 4,
                        Qt.GlobalColor.white,
                    )
                painter.fillRect(x, y, cell_width - 2, cell_height - 2, color)
                painter.setPen(
                    Qt.GlobalColor.white if intensity > 0.5 else Qt.GlobalColor.black
                )
                painter.setFont(QFont("Arial", 9))
                painter.drawText(
                    QRect(x, y, cell_width - 2, cell_height - 2),
                    Qt.AlignmentFlag.AlignCenter,
                    f"{value:.1f}",
                )
        painter.setPen(Qt.GlobalColor.white)
        painter.setFont(QFont("Arial", 10))
        for i, label in enumerate(self.row_labels):
            y = margin + i * cell_height + cell_height // 2
            painter.drawText(
                QRect(0, y - 10, margin - 5, 20),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
        for j, label in enumerate(self.col_labels):
            x = margin + j * cell_width + cell_width // 2
            painter.save()
            painter.translate(x, margin - 5)
            painter.rotate(-45)
            painter.drawText(
                QRect(-50, -20, 100, 20), Qt.AlignmentFlag.AlignCenter, label
            )
            painter.restore()


# ==================== DASHBOARD WIDGET ====================
class AnalyticsDashboardWidget(QWidget):
    """Complete analytics dashboard combining multiple charts"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setupUI()
        self.loadDemoData()

    def setupUI(self) -> None:
        layout = QGridLayout(self)

        title = QLabel("Legal AI Analytics Dashboard")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #dc143c;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title, 0, 0, 1, 2)

        self.pie_chart = PieChartWidget()
        self.pie_chart.segmentClicked.connect(
            lambda label, value: print(f"Pie segment clicked: {label} = {value}")
        )
        layout.addWidget(self.pie_chart, 1, 0)

        self.bar_chart = BarChartWidget()
        self.bar_chart.barClicked.connect(
            lambda label, value: print(f"Bar clicked: {label} = {value}")
        )
        layout.addWidget(self.bar_chart, 1, 1)

        self.line_chart = LineChartWidget()
        layout.addWidget(self.line_chart, 2, 0, 1, 2)

        self.heat_map = HeatMapWidget()
        layout.addWidget(self.heat_map, 3, 0, 1, 2)

    def loadDemoData(self) -> None:
        pie_data = [
            ChartData("Contracts", 450),
            ChartData("Legal Briefs", 320),
            ChartData("Patents", 180),
            ChartData("Compliance", 150),
            ChartData("Other", 100),
        ]
        self.pie_chart.setData(pie_data)

        bar_data = [
            ChartData("Monday", 45),
            ChartData("Tuesday", 38),
            ChartData("Wednesday", 52),
            ChartData("Thursday", 41),
            ChartData("Friday", 35),
        ]
        self.bar_chart.setData(bar_data)

        now = datetime.now()
        success_data: list[tuple[datetime, float]] = []
        error_data: list[tuple[datetime, float]] = []
        for i in range(30):
            timestamp = now - timedelta(days=30 - i)
            success_rate = 85 + random.uniform(-10, 10)
            error_rate = 5 + random.uniform(-3, 3)
            success_data.append((timestamp, success_rate))
            error_data.append((timestamp, error_rate))
        self.line_chart.addSeries("Success Rate", success_data, QColor("#4caf50"))
        self.line_chart.addSeries("Error Rate", error_data, QColor("#f44336"))

        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        hours = [f"{i:02d}:00" for i in range(9, 18)]
        heat_data = []
        for day in range(7):
            row = []
            for hour in range(9):
                if day < 5 and 2 < hour < 7:
                    activity = random.uniform(0.6, 1.0)
                else:
                    activity = random.uniform(0.0, 0.3)
                row.append(activity)
            heat_data.append(row)
        self.heat_map.setData(heat_data, days, hours)
