from __future__ import annotations

"""Document handling helpers for the integrated GUI."""

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

from PyQt6.QtCore import QModelIndex, QPoint, Qt
from PyQt6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMenu,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)

from ..legal_ai_desktop import DocumentViewer, DocumentTableModel, Document
from ..legal_ai_widgets import NotificationWidget, SearchableComboBox


class DocumentSection:
    """Manage document-related views and actions."""

    def __init__(self, main_window: QWidget) -> None:
        self.main_window = main_window
        self.db_manager = main_window.db_manager
        self.api_client = main_window.api_client
        self.search_engine = main_window.search_engine
        self.processing_worker = main_window.processing_worker

        self.documents: Dict[str, Document] = {}
        self.active_viewers: List[DocumentViewer] = []
        self.doc_model = DocumentTableModel()
        self.widget: QWidget | None = None
        self.status_filter: SearchableComboBox | None = None
        self.type_filter: SearchableComboBox | None = None
        self.doc_table: QTableView | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def create_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)

        toolbar_layout.addWidget(QLabel("Status:"))
        self.status_filter = SearchableComboBox()
        self.status_filter.addItems([
            "All",
            "Pending",
            "Processing",
            "Completed",
            "Failed",
        ])
        self.status_filter.currentTextChanged.connect(self.filter_documents)
        toolbar_layout.addWidget(self.status_filter)

        toolbar_layout.addWidget(QLabel("Type:"))
        self.type_filter = SearchableComboBox()
        self.type_filter.addItems(
            ["All", "Contract", "Legal Brief", "Patent", "Compliance"]
        )
        toolbar_layout.addWidget(self.type_filter)

        toolbar_layout.addStretch()

        export_btn = QPushButton("Export Selected")
        export_btn.clicked.connect(self.export_documents)
        toolbar_layout.addWidget(export_btn)

        layout.addWidget(toolbar)

        self.doc_table = QTableView()
        self.doc_table.setModel(self.doc_model)
        self.doc_table.setSelectionBehavior(
            QTableView.SelectionBehavior.SelectRows
        )
        self.doc_table.setAlternatingRowColors(True)
        self.doc_table.doubleClicked.connect(self.view_document)
        self.doc_table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.doc_table.customContextMenuRequested.connect(
            self.show_context_menu
        )
        layout.addWidget(self.doc_table)

        self.widget = widget
        return widget

    # ------------------------------------------------------------------
    # Document utilities
    # ------------------------------------------------------------------
    def load_local_documents(self) -> None:
        docs = self.db_manager.getDocuments(limit=1000)
        for doc_data in docs:
            doc = Document(
                id=doc_data["document_id"],
                filename=doc_data["filename"],
                status=doc_data["status"],
                progress=1.0 if doc_data["status"] == "completed" else 0.5,
                uploaded_at=datetime.fromisoformat(doc_data["upload_date"]),
                file_size=doc_data.get("file_size", 0),
            )
            self.documents[doc.id] = doc
            self.doc_model.addDocument(doc)

    def load_documents(self) -> None:
        self.api_client.loadDocuments()

    def upload_documents(self) -> None:
        files, _ = self.main_window.QFileDialog.getOpenFileNames(
            self.main_window,
            "Select Documents",
            str(Path.home()),
            "Documents (*.pdf *.docx *.txt *.md);;All Files (*.*)",
        )
        if files:
            self.main_window.upload_btn.startGlow()
            for file_path in files:
                path = Path(file_path)
                self.api_client.uploadDocument(
                    path,
                    {
                        "enable_ner": self.main_window.prefs_manager.get(
                            "enable_ner", True
                        ),
                        "enable_llm": self.main_window.prefs_manager.get(
                            "enable_llm", True
                        ),
                        "confidence_threshold": self.main_window.prefs_manager.get(
                            "confidence_threshold", 0.7
                        ),
                    },
                )
            self.main_window.upload_btn.stopGlow()

    def view_document(self, index: QModelIndex) -> None:
        row = index.row()
        doc_id = self.doc_model.documents.iloc[row]["ID"]
        viewer = DocumentViewer(doc_id, self.main_window)
        viewer.show()
        self.active_viewers.append(viewer)

    def perform_global_search(self) -> None:
        query: str = self.main_window.global_search.text()
        if query:
            results = self.search_engine.search(query)
            self.main_window.log(f"Search returned {len(results)} results")
            results_widget = QListWidget()
            for result in results:
                item = QListWidgetItem(
                    f"{result['filename']}: {result['snippet']}"
                )
                item.setData(Qt.ItemDataRole.UserRole, result["document_id"])
                results_widget.addItem(item)
            self.main_window.main_tabs.addTab(
                results_widget, f"Search: {query}"
            )
            self.main_window.main_tabs.setCurrentWidget(results_widget)

    # ------------------------------------------------------------------
    # Context menu and filtering
    # ------------------------------------------------------------------
    def filter_documents(self) -> None:  # pragma: no cover - GUI method
        pass

    def export_documents(self) -> None:  # pragma: no cover - GUI method
        pass

    def show_context_menu(self, pos: QPoint) -> None:
        menu = QMenu(self.doc_table)
        view_action = menu.addAction("View")
        view_action.triggered.connect(
            lambda: self.view_document(self.doc_table.currentIndex())
        )
        menu.addSeparator()
        menu.addAction("Reprocess")
        menu.addAction("Delete")
        menu.exec(self.doc_table.mapToGlobal(pos))

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------
    def on_documents_loaded(self, documents: List[Dict[str, Any]]) -> None:
        self.main_window.log(f"Loaded {len(documents)} documents from server")
        for doc_data in documents:
            doc = Document(
                id=doc_data["document_id"],
                filename=doc_data["filename"],
                status=doc_data["status"],
                progress=doc_data.get("progress", 0),
                uploaded_at=datetime.fromisoformat(doc_data["uploaded_at"]),
                file_size=doc_data.get("file_size", 0),
                doc_type=doc_data.get("type", "Unknown"),
            )
            self.documents[doc.id] = doc
            self.doc_model.addDocument(doc)
            self.db_manager.saveDocument(
                doc.id,
                doc.filename,
                file_size=doc.file_size,
                metadata=doc_data.get("metadata"),
            )

    def on_document_uploaded(self, doc_id: str) -> None:
        self.main_window.log(f"Document uploaded: {doc_id}")
        NotificationWidget("Document uploaded successfully", "success").show(
            self.main_window
        )
        if doc_id in self.documents:
            self.main_window.queue_section.queue_list.addItem(
                f"Processing: {self.documents[doc_id].filename}"
            )

    def on_processing_complete(self, doc_id: str, results: Dict[str, Any]) -> None:
        self.main_window.log(f"Processing complete for document: {doc_id}")
        if doc_id in self.documents:
            self.documents[doc_id].status = "completed"
            self.doc_model.updateDocument(doc_id, "completed", 1.0)
        self.db_manager.updateDocumentStatus(doc_id, "completed", results)
        if "text_content" in results:
            self.search_engine.indexDocument(
                doc_id,
                self.documents[doc_id].filename,
                results["text_content"],
                results,
            )
        NotificationWidget(
            f"Processing complete: {self.documents[doc_id].filename}", "success"
        ).show(self.main_window)

    def on_processing_progress(self, doc_id: str, progress: int, stage: str) -> None:
        self.main_window.log(f"Processing {doc_id}: {stage} ({progress}%)")
        if doc_id in self.documents:
            self.doc_model.updateDocument(doc_id, "processing", progress / 100)

