"""Minimal desktop UI components for the integrated GUI."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QTableView,
)
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex
import pandas as pd

from legal_ai_system.gui.legal_ai_charts import AnalyticsDashboardWidget


@dataclass
class Document:
    """Simple document structure used by the GUI."""

    id: str
    filename: str
    status: str = "pending"
    progress: float = 0.0
    uploaded_at: datetime | None = None
    file_size: int = 0
    doc_type: str = "Unknown"


class LegalAIApplication(QApplication):
    """Thin wrapper around :class:`QApplication`."""


class MainWindow(QMainWindow):
    """Base main window placeholder."""


class SettingsDialog(QDialog):
    """Basic settings dialog with a few editable fields."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        layout = QVBoxLayout(self)
        self.api_url = QLabel("API URL: placeholder")
        self.openai_key = QLabel("OpenAI Key: ********")
        self.enable_ner = QLabel("NER Enabled")
        self.enable_llm = QLabel("LLM Enabled")
        layout.addWidget(self.api_url)
        layout.addWidget(self.openai_key)
        layout.addWidget(self.enable_ner)
        layout.addWidget(self.enable_llm)


class AboutDialog(QDialog):
    """Simple about dialog."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Legal AI System"))


class DocumentViewer(QWidget):
    """Trivial document viewer."""

    def __init__(self, doc_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Document {doc_id}")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Viewing document: {doc_id}"))


class DocumentTableModel(QAbstractTableModel):
    """Very small table model for storing documents."""

    headers = [
        "ID",
        "Filename",
        "Status",
        "Progress",
        "UploadedAt",
        "Size",
        "Type",
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.documents = pd.DataFrame(columns=self.headers)

    # Qt model implementation -------------------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: D401
        return 0 if parent.isValid() else len(self.documents)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: D401
        return 0 if parent.isValid() else len(self.documents.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        value = self.documents.iat[index.row(), index.column()]
        return str(value)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            try:
                return str(self.documents.columns[section])
            except IndexError:
                return None
        return str(section)

    # Convenience helpers ----------------------------------------------------
    def addDocument(self, doc: Document) -> None:
        row = {
            "ID": doc.id,
            "Filename": doc.filename,
            "Status": doc.status,
            "Progress": doc.progress,
            "UploadedAt": doc.uploaded_at.isoformat() if doc.uploaded_at else "",
            "Size": doc.file_size,
            "Type": doc.doc_type,
        }
        self.beginInsertRows(QModelIndex(), len(self.documents), len(self.documents))
        self.documents = pd.concat([self.documents, pd.DataFrame([row])], ignore_index=True)
        self.endInsertRows()

    def updateDocument(self, doc_id: str, status: str, progress: float) -> None:
        idx = self.documents.index[self.documents["ID"] == doc_id]
        if not idx.empty:
            row = idx[0]
            self.documents.at[row, "Status"] = status
            self.documents.at[row, "Progress"] = progress
            self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount() - 1))


# Expose AnalyticsDashboard as alias to the widget implementation
AnalyticsDashboard = AnalyticsDashboardWidget

__all__ = [
    "MainWindow",
    "DocumentViewer",
    "AnalyticsDashboard",
    "SettingsDialog",
    "AboutDialog",
    "LegalAIApplication",
    "DocumentTableModel",
    "Document",
]
