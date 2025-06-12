"""Drag-and-drop workflow builder widgets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from PyQt6.QtCore import Qt, QMimeData, QPoint
from PyQt6.QtGui import QDrag, QMouseEvent
from PyQt6.QtWidgets import (
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QFrame,
)


class DraggableComponentButton(QPushButton):
    """Button representing a workflow component that can be dragged."""

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(name, parent)
        self._drag_start: QPoint | None = None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            event.buttons() & Qt.MouseButton.LeftButton
            and self._drag_start is not None
            and (event.pos() - self._drag_start).manhattanLength() > 10
        ):
            mime = QMimeData()
            mime.setText(self.text())
            drag = QDrag(self)
            drag.setMimeData(mime)
            drag.exec(Qt.DropAction.CopyAction)
        super().mouseMoveEvent(event)


@dataclass
class WorkflowCanvas(QFrame):
    """Canvas accepting components via drag-and-drop."""

    components: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.Shape.Box)
        self._layout = QVBoxLayout(self)
        self._layout.addStretch()
        self._update_view()

    # Visual feedback -------------------------------------------------
    def dragEnterEvent(self, event) -> None:  # pragma: no cover - UI
        if event.mimeData().hasText():
            event.acceptProposedAction()
            self.setStyleSheet("border: 2px solid #4caf50;")
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # pragma: no cover - UI
        self.setStyleSheet("")
        event.accept()

    def dropEvent(self, event) -> None:  # pragma: no cover - UI
        if event.mimeData().hasText():
            name = event.mimeData().text()
            self.components.append(name)
            self._update_view()
            self.setStyleSheet("")
            event.acceptProposedAction()
        else:
            event.ignore()

    # Workflow operations ----------------------------------------------
    def _update_view(self) -> None:
        # Remove old labels except stretch
        for i in reversed(range(self._layout.count() - 1)):
            item = self._layout.takeAt(i)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        for idx, comp in enumerate(self.components, 1):
            lbl = QLabel(f"{idx}. {comp}")
            self._layout.insertWidget(idx - 1, lbl)

    def save_workflow(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.components, fh, indent=2)

    def load_workflow(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            self.components = json.load(fh)
        self._update_view()

    @property
    def component_count(self) -> int:
        return len(self.components)

    def validate_workflow(self) -> bool:
        return self.component_count > 0

    def clear(self) -> None:
        self.components.clear()
        self._update_view()


class WorkflowBuilderWidget(QWidget):
    """Composite widget with component palette and canvas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.canvas = WorkflowCanvas()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)

        # Component palette
        palette = QVBoxLayout()
        for name in ["Loader", "Analyzer", "Summarizer", "Exporter"]:
            palette.addWidget(DraggableComponentButton(name))
        palette.addStretch()
        layout.addLayout(palette)

        layout.addWidget(self.canvas, stretch=1)

        # Control buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        load_btn = QPushButton("Load")
        validate_btn = QPushButton("Validate")
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(validate_btn)
        layout.addLayout(btn_layout)

        save_btn.clicked.connect(self._save)
        load_btn.clicked.connect(self._load)
        validate_btn.clicked.connect(self._validate)

    def _save(self) -> None:  # pragma: no cover - UI
        path, _ = QFileDialog.getSaveFileName(self, "Save Workflow", "workflow.json", "JSON Files (*.json)")
        if path:
            self.canvas.save_workflow(path)

    def _load(self) -> None:  # pragma: no cover - UI
        path, _ = QFileDialog.getOpenFileName(self, "Load Workflow", "", "JSON Files (*.json)")
        if path:
            self.canvas.load_workflow(path)

    def _validate(self) -> None:  # pragma: no cover - UI
        msg = "Valid" if self.canvas.validate_workflow() else "Invalid"
        QLabel(msg).show()


__all__ = [
    "DraggableComponentButton",
    "WorkflowCanvas",
    "WorkflowBuilderWidget",
]

