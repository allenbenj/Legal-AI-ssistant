"""Simplified PyQt6 interface for the Legal AI System.

This implementation provides a basic document browser using
:class:`DatabaseManager` for local storage. It replaces missing modules
from the original prototype with lightweight placeholders so the GUI
can run standalone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets

from ..services.database_manager import DatabaseManager


@dataclass
class Document:
    """Simple document representation for the table view."""

    id: str
    filename: str
    status: str = "pending"
    progress: float = 0.0
    uploaded_at: datetime = field(default_factory=datetime.utcnow)
    file_size: int = 0
    doc_type: str = "Unknown"


class DocumentTableModel(QtCore.QAbstractTableModel):
    """Table model backed by a ``pandas`` DataFrame."""

    headers = ["ID", "Filename", "Status", "Progress", "Type"]

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.documents = pd.DataFrame(columns=self.headers)

    # -- Qt model implementation -------------------------------------------------
    def rowCount(self, parent: QtCore.QModelIndex | QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.documents)

    def columnCount(self, parent: QtCore.QModelIndex | QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.headers)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        value = self.documents.iat[index.row(), index.column()]
        return str(value)

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> Any:
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            return self.headers[section]
        return str(section)

    # -- convenience helpers -----------------------------------------------------
    def add_document(self, doc: Document) -> None:
        row = {
            "ID": doc.id,
            "Filename": doc.filename,
            "Status": doc.status,
            "Progress": f"{int(doc.progress * 100)}%",
            "Type": doc.doc_type,
        }
        self.beginInsertRows(QtCore.QModelIndex(), len(self.documents), len(self.documents))
        self.documents = pd.concat([self.documents, pd.DataFrame([row])], ignore_index=True)
        self.endInsertRows()

    def update_document(self, doc_id: str, status: str, progress: float) -> None:
        idx_list = self.documents.index[self.documents["ID"] == doc_id].tolist()
        if not idx_list:
            return
        row = idx_list[0]
        self.documents.at[row, "Status"] = status
        self.documents.at[row, "Progress"] = f"{int(progress * 100)}%"
        top_left = self.index(row, 0)
        bottom_right = self.index(row, len(self.headers) - 1)
        self.dataChanged.emit(top_left, bottom_right)


class PreferencesManager(QtCore.QObject):
    """Wrapper around ``QSettings`` with change notification."""

    preferenceChanged = QtCore.pyqtSignal(str, object)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._settings = QtCore.QSettings("LegalAI", "Desktop")

    def get(self, key: str, default: Any = None) -> Any:
        return self._settings.value(key, default)

    def set(self, key: str, value: Any) -> None:
        self._settings.setValue(key, value)
        self.preferenceChanged.emit(key, value)


class IntegratedMainWindow(QtWidgets.QMainWindow):
    """Main window integrating basic document management."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Legal AI System")
        self.resize(1200, 800)

        self.db_manager = DatabaseManager()
        self.prefs_manager = PreferencesManager(self)

        self.documents: Dict[str, Document] = {}
        self.doc_model = DocumentTableModel()

        self._build_ui()
        self._load_documents()

    # -- UI construction ---------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Toolbar with search and actions
        toolbar = QtWidgets.QHBoxLayout()
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Search documents...")
        self.search_edit.returnPressed.connect(self._search)
        toolbar.addWidget(self.search_edit)

        upload_btn = QtWidgets.QPushButton("Upload")
        upload_btn.clicked.connect(self._upload)
        toolbar.addWidget(upload_btn)

        layout.addLayout(toolbar)

        # Document table
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.doc_model)
        self.table.setSelectionBehavior(QtWidgets.QTableView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

    # -- database helpers --------------------------------------------------------
    def _load_documents(self) -> None:
        """Load records from the local database into the table model."""
        try:
            records = self.db_manager.get_documents(limit=1000)
        except Exception as exc:  # pragma: no cover - sqlite errors are unlikely
            QtWidgets.QMessageBox.critical(self, "Database Error", str(exc))
            return
        for rec in records:
            doc = Document(
                id=rec["id"],
                filename=rec["filename"],
                status=rec["processing_status"],
                progress=1.0 if rec["processing_status"] == "completed" else 0.0,
                uploaded_at=datetime.fromisoformat(rec["upload_time"]),
                file_size=rec["file_size"],
                doc_type=rec["file_type"],
            )
            self.documents[doc.id] = doc
            self.doc_model.add_document(doc)

    # -- actions ----------------------------------------------------------------
    def _upload(self) -> None:
        """Add new documents to the database and table."""
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Documents", str(Path.home()), "Documents (*.pdf *.txt *.docx)"
        )
        for file_path in files:
            path = Path(file_path)
            doc_id = path.stem
            self.db_manager.save_document(
                doc_id, path.name, path.suffix.lstrip("."), path.stat().st_size, {}
            )
            doc = Document(
                id=doc_id,
                filename=path.name,
                status="uploaded",
                progress=0.0,
                uploaded_at=datetime.utcnow(),
                file_size=path.stat().st_size,
                doc_type=path.suffix.lstrip("."),
            )
            self.documents[doc.id] = doc
            self.doc_model.add_document(doc)

    def _search(self) -> None:
        query = self.search_edit.text().lower()
        if not query:
            return
        results = [d for d in self.documents.values() if query in d.filename.lower()]
        QtWidgets.QMessageBox.information(self, "Search", f"Found {len(results)} document(s)")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = IntegratedMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
