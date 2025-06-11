from __future__ import annotations

"""PyQt6 widget implementing the Memory Brain panel."""

import asyncio
import json
from typing import Any, Dict, List

from PyQt6 import QtCore, QtWidgets

from .panels.memory_brain_panel import MemoryEntry, ContradictionDetector
from ..services.memory_service import memory_manager_context


class MemoryBrainWidget(QtWidgets.QWidget):
    """Widget providing statement intake and contradiction checks."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.memory_entries: List[MemoryEntry] = []
        self._load_memory_entries()
        self._build_ui()

    # ------------------------------------------------------------------
    # Data management helpers
    # ------------------------------------------------------------------
    def _load_memory_entries(self) -> None:
        async def _load() -> None:
            async with memory_manager_context() as manager:
                entries = await manager.get_context_window("memory_brain")
                self.memory_entries = [
                    MemoryEntry(
                        speaker=e.get("metadata", {}).get("speaker", ""),
                        statement=e.get("content", ""),
                        source=e.get("metadata", {}).get("source", ""),
                    )
                    for e in entries
                    if e.get("entry_type") == "statement"
                ]

        try:
            asyncio.run(_load())
        except Exception:  # pragma: no cover - initialization can fail offline
            self.memory_entries = []

    def _persist_statement(self, entry: MemoryEntry) -> None:
        async def _store() -> None:
            async with memory_manager_context() as manager:
                await manager.add_context_window_entry(
                    session_id="memory_brain",
                    entry_type="statement",
                    content=entry.statement,
                    metadata={"speaker": entry.speaker, "source": entry.source},
                )

        try:
            asyncio.run(_store())
        except Exception:  # pragma: no cover - storage errors are non fatal
            pass

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_intake_tab(), "Statement Intake")
        tabs.addTab(self._build_check_tab(), "Contradiction Check")
        tabs.addTab(self._build_curate_tab(), "Merge & Curate")

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
            self.memory_entries.append(entry)
            self._persist_statement(entry)
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
            detector = ContradictionDetector(self.memory_entries)
            result = detector.check(
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
            "Additional curation tools would appear here.", alignment=QtCore.Qt.AlignmentFlag.AlignTop
        )
        layout = QtWidgets.QVBoxLayout(widget)
        layout.addWidget(label)
        return widget


__all__ = ["MemoryBrainWidget"]
