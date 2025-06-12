from __future__ import annotations

"""Processing queue helpers for the integrated GUI."""

from pathlib import Path
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class QueueSection:
    """Manage processing queue UI and operations."""

    def __init__(self, main_window: QWidget) -> None:
        self.main_window = main_window
        self.processing_worker = main_window.processing_worker
        self.queue_list: QListWidget | None = None
        self.widget: QWidget | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def create_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)

        pause_btn = QPushButton("Pause Queue")
        pause_btn.setCheckable(True)
        pause_btn.toggled.connect(self.toggle_processing)
        controls_layout.addWidget(pause_btn)

        clear_btn = QPushButton("Clear Completed")
        clear_btn.clicked.connect(self.clear_completed)
        controls_layout.addWidget(clear_btn)

        controls_layout.addStretch()
        layout.addWidget(controls)

        self.queue_list = QListWidget()
        self.queue_list.setAlternatingRowColors(True)
        layout.addWidget(self.queue_list)

        self.widget = widget
        return widget

    # ------------------------------------------------------------------
    # Queue actions
    # ------------------------------------------------------------------
    def process_queue(self) -> None:
        for doc_id, doc in self.main_window.document_section.documents.items():
            if doc.status == "pending":
                self.processing_worker.addDocument(
                    doc_id,
                    Path(doc.filename),
                    {},
                )

    def toggle_processing(self, paused: bool) -> None:
        if paused:
            self.processing_worker.requestInterruption()
        else:
            self.process_queue()

    def clear_completed(self) -> None:
        if not self.queue_list:
            return
        for i in range(self.queue_list.count() - 1, -1, -1):
            item = self.queue_list.item(i)
            if "Complete" in item.text():
                self.queue_list.takeItem(i)
