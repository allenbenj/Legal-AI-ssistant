"""Extra PyQt6 windows for document viewing and analytics."""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from .workflow_progress_widget import WorkflowProgressWidget


class DocumentViewerWindow(QtWidgets.QMainWindow):
    """Simple window displaying a document with animated progress."""

    def __init__(self, ws_base_url: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Document Viewer")
        self.resize(600, 400)

        self.text_edit = QtWidgets.QTextEdit(readOnly=True)
        self.progress_widget = WorkflowProgressWidget(ws_base_url)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.progress_widget)
        self.setCentralWidget(central)

        self.progress_widget.start()

    def load_file(self, path: str) -> None:
        try:
            content = Path(path).read_text()
        except Exception:
            content = "Unable to load file"
        self.text_edit.setPlainText(content)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: D401
        self.progress_widget.stop()
        super().closeEvent(event)


class AnalyticsWindow(QtWidgets.QMainWindow):
    """Display analytics information with animated indicator."""

    def __init__(self, ws_base_url: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Analytics")
        self.resize(400, 200)

        self.progress_widget = WorkflowProgressWidget(ws_base_url)
        self.indicator = QtWidgets.QFrame()
        self.indicator.setFixedHeight(10)
        self.indicator.setStyleSheet("background-color: #448aff;")

        self.color_anim = QtCore.QPropertyAnimation(self.indicator, b"styleSheet", self)
        self.color_anim.setDuration(500)
        self.color_anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(self.progress_widget)
        layout.addWidget(self.indicator)
        self.setCentralWidget(central)

        self.progress_widget.start()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: D401
        self.progress_widget.stop()
        super().closeEvent(event)

    def animate_indicator(self) -> None:
        """Pulse the indicator color to draw attention."""
        self.color_anim.stop()
        self.color_anim.setStartValue("background-color: #448aff;")
        self.color_anim.setEndValue("background-color: #82b1ff;")
        self.color_anim.start()


__all__ = ["DocumentViewerWindow", "AnalyticsWindow"]
