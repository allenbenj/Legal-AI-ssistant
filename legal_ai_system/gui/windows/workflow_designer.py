from __future__ import annotations

"""Simplistic workflow designer browser."""

from PyQt6 import QtWidgets

from ..main_gui import APIClient


class WorkflowDesignerWindow(QtWidgets.QMainWindow):
    """Display available workflows."""

    def __init__(self, api_client: APIClient) -> None:
        super().__init__()
        self.api_client = api_client
        self.setWindowTitle("Workflow Designer")

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["ID", "Name"])
        self.table.horizontalHeader().setStretchLastSection(True)

        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_workflows)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(refresh_btn)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.load_workflows()

    def load_workflows(self) -> None:
        workflows = self.api_client.workflows()
        self.table.setRowCount(len(workflows))
        for row, wf in enumerate(workflows):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(wf.get("id", "")))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(wf.get("name", "")))


__all__ = ["WorkflowDesignerWindow"]
