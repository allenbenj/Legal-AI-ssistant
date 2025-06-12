from __future__ import annotations

"""Window for browsing processed documents."""

from PyQt6 import QtWidgets

from ..main_gui import APIClient
from .table_models import DocumentTableModel


class DocumentViewerWindow(QtWidgets.QMainWindow):
    """Display documents returned by the backend."""

    def __init__(self, api_client: APIClient) -> None:
        super().__init__()
        self.api_client = api_client
        self.setWindowTitle("Document Viewer")

        self.model = DocumentTableModel(api_client)
        self.view = QtWidgets.QTableView()
        self.view.setModel(self.model)
        self.view.setSortingEnabled(True)

        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Filter...")
        self.filter_edit.textChanged.connect(self.model.filter_keyword)

        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self.model.fetch)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.filter_edit)
        top.addWidget(refresh_btn)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addLayout(top)
        layout.addWidget(self.view)
        self.setCentralWidget(central)

        self.model.fetch()


__all__ = ["DocumentViewerWindow"]
