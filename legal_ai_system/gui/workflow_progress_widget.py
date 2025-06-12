"""PyQt6 widget showing workflow progress with animated bar."""

from __future__ import annotations

import uuid

from PyQt6 import QtCore, QtWidgets

from .main_gui import WebSocketWorker


class WorkflowProgressWidget(QtWidgets.QWidget):
    """Display workflow progress updates from a WebSocket connection."""

    def __init__(self, ws_base_url: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.ws_base = ws_base_url.rstrip("/")
        self.client_id = uuid.uuid4().hex

        self.progress = QtWidgets.QProgressBar()
        self.stage_label = QtWidgets.QLabel()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.progress)

        self.animation = QtCore.QPropertyAnimation(self.progress, b"value", self)
        self.animation.setDuration(400)
        self.animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)

        self.ws_worker: WebSocketWorker | None = None

    def start(self) -> None:
        """Begin listening for workflow progress updates."""
        url = f"{self.ws_base}/ws/{self.client_id}"
        topics = ["workflow_progress"]
        self.ws_worker = WebSocketWorker(url, topics)
        self.ws_worker.message_received.connect(self.handle_update)
        self.ws_worker.start()

    def stop(self) -> None:
        if self.ws_worker:
            self.ws_worker.stop()
            self.ws_worker.wait(1000)
            self.ws_worker = None

    def handle_update(self, data: dict) -> None:
        if data.get("type") != "workflow_progress":
            return
        progress = int(float(data.get("progress", 0)) * 100)
        stage = data.get("message") or data.get("stage", "")
        self.stage_label.setText(stage)
        self.animate_to(progress)

    def animate_to(self, value: int) -> None:
        self.animation.stop()
        self.animation.setStartValue(self.progress.value())
        self.animation.setEndValue(value)
        self.animation.start()


__all__ = ["WorkflowProgressWidget"]
