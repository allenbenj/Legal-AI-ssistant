"""Dialog for configuring database connection information.

This standalone dialog detects available SQLite databases within the
project directory and allows the user to enter connection details for a
remote database server. Values are persisted using ``QSettings`` so they
are restored across sessions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QVBoxLayout,
)


class DBConnectionDialog(QDialog):
    """Dialog to configure database connections."""

    def __init__(self, parent: QDialog | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Database Connections")
        self.resize(500, 400)

        main_layout = QVBoxLayout(self)

        main_layout.addWidget(QLabel("Detected Databases:"))
        self.available_list = QListWidget()
        main_layout.addWidget(self.available_list)

        form_layout = QFormLayout()
        self.db_type_combo = QComboBox()
        self.db_type_combo.addItems(["PostgreSQL", "SQLite"])
        form_layout.addRow("Type:", self.db_type_combo)
        self.host_edit = QLineEdit()
        self.port_edit = QLineEdit()
        self.user_edit = QLineEdit()
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.db_name_edit = QLineEdit()
        form_layout.addRow("Host:", self.host_edit)
        form_layout.addRow("Port:", self.port_edit)
        form_layout.addRow("User:", self.user_edit)
        form_layout.addRow("Password:", self.password_edit)
        form_layout.addRow("Database:", self.db_name_edit)
        main_layout.addLayout(form_layout)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        main_layout.addWidget(btn_box)

        self.load_settings()
        self.populate_databases()

    # ------------------------------------------------------------------
    def populate_databases(self) -> None:
        """Populate the list of available SQLite databases."""
        self.available_list.clear()
        for path in Path.cwd().rglob("*.db"):
            self.available_list.addItem(str(path))

    def load_settings(self) -> None:
        settings = QSettings("LegalAI", "Desktop")
        self.db_type_combo.setCurrentText(settings.value("db/type", "PostgreSQL"))
        self.host_edit.setText(settings.value("db/host", "localhost"))
        self.port_edit.setText(settings.value("db/port", "5432"))
        self.user_edit.setText(settings.value("db/user", ""))
        self.password_edit.setText(settings.value("db/password", ""))
        self.db_name_edit.setText(settings.value("db/name", "legal_ai"))

    def save_settings(self) -> None:
        settings = QSettings("LegalAI", "Desktop")
        settings.setValue("db/type", self.db_type_combo.currentText())
        settings.setValue("db/host", self.host_edit.text())
        settings.setValue("db/port", self.port_edit.text())
        settings.setValue("db/user", self.user_edit.text())
        settings.setValue("db/password", self.password_edit.text())
        settings.setValue("db/name", self.db_name_edit.text())

    def accept(self) -> None:  # type: ignore[override]
        self.save_settings()
        if self.db_type_combo.currentText() == "PostgreSQL":
            import os

            os.environ["POSTGRES_HOST"] = self.host_edit.text()
            os.environ["POSTGRES_PORT"] = self.port_edit.text()
            os.environ["POSTGRES_USER"] = self.user_edit.text()
            os.environ["POSTGRES_PASSWORD"] = self.password_edit.text()
            os.environ["POSTGRES_DB"] = self.db_name_edit.text()
            os.environ["DATABASE_URL"] = (
                f"postgresql://{self.user_edit.text()}:{self.password_edit.text()}@"
                f"{self.host_edit.text()}:{self.port_edit.text()}/{self.db_name_edit.text()}"
            )
        super().accept()


__all__: Iterable[str] = ["DBConnectionDialog"]
