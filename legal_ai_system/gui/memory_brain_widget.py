from __future__ import annotations

"""PyQt6 widget implementing the Memory Brain panel."""

import json
from typing import Any, Dict, List

from PyQt6 import QtCore, QtWidgets

from .memory_table_model import MemoryTableWidget

from .panels.memory_brain_panel import MemoryEntry
from .memory_brain_core import MemoryBrainCore


class MemoryBrainWidget(QtWidgets.QWidget):
    """Widget providing statement intake and contradiction checks."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.core = MemoryBrainCore()
        self.core.load_entries()
        self.memory_entries = self.core.memory_entries
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_intake_tab(), "Statement Intake")
        tabs.addTab(self._build_check_tab(), "Contradiction Check")
        tabs.addTab(self._build_curate_tab(), "Merge & Curate")
        tabs.addTab(self._build_view_tab(), "Memory View")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(tabs)

    def _build_intake_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        speaker_edit = QtWidgets.QLineEdit()
        statement_edit = QtWidgets.QTextEdit()
        source_edit = QtWidgets.QLineEdit()
        add_btn = QtWidgets.QPushButton("Add Statement")
        notice = QtWidgets.QLabel()

        form = QtWidgets.QFormLayout(widget)
        form.addRow("Speaker", speaker_edit)
        form.addRow("Statement", statement_edit)
        form.addRow("Source", source_edit)
        form.addRow(add_btn)
        form.addRow(notice)

        def on_add() -> None:
            entry = MemoryEntry(
                speaker=speaker_edit.text(),
                statement=statement_edit.toPlainText(),
                source=source_edit.text(),
            )
            self.core.add_statement(entry)
            notice.setText("Statement added to memory.")
            speaker_edit.clear()
            statement_edit.clear()
            source_edit.clear()

        add_btn.clicked.connect(on_add)
        return widget

    def _build_check_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        speaker_edit = QtWidgets.QLineEdit()
        statement_edit = QtWidgets.QTextEdit()
        source_edit = QtWidgets.QLineEdit()
        run_btn = QtWidgets.QPushButton("Run Check")
        output = QtWidgets.QTextEdit(readOnly=True)

        form = QtWidgets.QFormLayout(widget)
        form.addRow("Speaker", speaker_edit)
        form.addRow("Statement", statement_edit)
        form.addRow("Source", source_edit)
        form.addRow(run_btn)
        form.addRow(output)

        def on_run() -> None:
            result = self.core.check(
                speaker_edit.text(), statement_edit.toPlainText(), source_edit.text()
            )
            msg = f"Contradictions found: {result['count']}"
            if result["contradictions"]:
                msg += "\n" + json.dumps(result["contradictions"], indent=2)
            output.setPlainText(msg)

        run_btn.clicked.connect(on_run)
        return widget

    def _build_curate_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        label = QtWidgets.QLabel(
            "Additional curation tools would appear here.",
            alignment=QtCore.Qt.AlignmentFlag.AlignTop,
        )
        layout = QtWidgets.QVBoxLayout(widget)
        layout.addWidget(label)
        return widget

    def _build_view_tab(self) -> QtWidgets.QWidget:
        return MemoryTableWidget(parent=self)


__all__ = ["MemoryBrainWidget"]
