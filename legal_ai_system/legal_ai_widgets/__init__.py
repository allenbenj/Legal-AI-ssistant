# AGENT_STUB
"""Small collection of custom Qt widgets used by the GUI."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
    QVBoxLayout,
)


class GlowingButton(QPushButton):
    """Button that toggles a simple glow effect."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._glowing = False

    def startGlow(self) -> None:
        self._glowing = True
        self.setStyleSheet("background-color: #ffa500;")

    def stopGlow(self) -> None:
        self._glowing = False
        self.setStyleSheet("")


class FlipCard(QWidget):
    """Widget showing front/back text when clicked."""

    def __init__(self, front: str, back: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.front = QLabel(front)
        self.back = QLabel(back)
        self.back.hide()
        layout = QVBoxLayout(self)
        layout.addWidget(self.front)
        layout.addWidget(self.back)
        self.front.mousePressEvent = self._toggle  # type: ignore[assignment]
        self.back.mousePressEvent = self._toggle  # type: ignore[assignment]

    def _toggle(self, event) -> None:  # pragma: no cover - UI method
        if self.front.isVisible():
            self.front.hide()
            self.back.show()
        else:
            self.back.hide()
            self.front.show()


class TagCloud(QWidget):  # pragma: no cover - demo widget
    """Very small placeholder tag cloud."""

    def __init__(self, tags: list[str] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.labels: list[QLabel] = []
        for tag in tags or []:
            lbl = QLabel(tag)
            self.labels.append(lbl)
            layout.addWidget(lbl)


class TimelineWidget(QWidget):  # pragma: no cover - demo widget
    """Placeholder timeline view."""


class NotificationWidget(QMessageBox):
    """Simple pop-up notification."""

    def __init__(self, message: str, level: str = "info", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setText(message)
        if level == "success":
            self.setIcon(QMessageBox.Icon.Information)
        elif level == "error":
            self.setIcon(QMessageBox.Icon.Critical)
        else:
            self.setIcon(QMessageBox.Icon.Warning)

    def show(self, parent: QWidget | None = None) -> None:  # type: ignore[override]
        if parent:
            super().show()
        else:
            super().exec()


class SearchableComboBox(QComboBox):  # pragma: no cover - minimal behaviour
    """Combo box with typing filter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().textEdited.connect(self._filter_items)

    def _filter_items(self, text: str) -> None:
        for i in range(self.count()):
            self.setRowHidden(i, text.lower() not in self.itemText(i).lower())


class DockablePanel(QDockWidget):  # pragma: no cover - stub
    """Simple dock widget."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)


__all__ = [
    "GlowingButton",
    "FlipCard",
    "TagCloud",
    "TimelineWidget",
    "NotificationWidget",
    "SearchableComboBox",
    "DockablePanel",
]
