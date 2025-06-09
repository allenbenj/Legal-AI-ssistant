"""Minimal PyQt6 GUI for running the LangGraph workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..log_setup import init_logging

from PyQt6 import QtWidgets

from ..workflows.langgraph_setup import build_graph
from ..utils.document_utils import extract_text


class MainWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Legal AI System")
        self.resize(600, 400)

        self.topic_edit = QtWidgets.QLineEdit(self)
        self.topic_edit.setPlaceholderText("Topic")

        self.open_btn = QtWidgets.QPushButton("Open Document", self)
        self.open_btn.clicked.connect(self.open_document)

        self.output = QtWidgets.QTextEdit(self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.topic_edit)
        layout.addWidget(self.open_btn)
        layout.addWidget(self.output)

    def open_document(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Document")
        if not file_path:
            return
        text = extract_text(Path(file_path))
        topic = self.topic_edit.text() or "default"
        graph = build_graph(topic)
        result = graph.run(text)
        self.output.setPlainText(str(result))


def main() -> None:
    init_logging()
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

__all__ = ["main", "MainWindow"]
