"""Startup configuration dialog for database and LLM settings."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class StartupConfigDialog(QDialog):
    """Collect database connection and LLM provider configuration."""

    def __init__(self, parent: QDialog | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Initial Configuration")
        self.resize(600, 400)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Database tab -----------------------------------------------------
        self.db_tab = QWidget()
        db_layout = QFormLayout(self.db_tab)
        self.db_type_combo = QComboBox()
        self.db_type_combo.addItems(["PostgreSQL", "SQLite"])
        db_layout.addRow("Type:", self.db_type_combo)
        self.host_edit = QLineEdit()
        self.port_edit = QLineEdit()
        self.user_edit = QLineEdit()
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.db_name_edit = QLineEdit()
        db_layout.addRow("Host:", self.host_edit)
        db_layout.addRow("Port:", self.port_edit)
        db_layout.addRow("User:", self.user_edit)
        db_layout.addRow("Password:", self.password_edit)
        db_layout.addRow("Database:", self.db_name_edit)
        self.available_list = QListWidget()
        db_layout.addRow(QLabel("Detected SQLite DBs:"), self.available_list)
        self.tabs.addTab(self.db_tab, "Database")

        # LLM tab ----------------------------------------------------------
        self.llm_tab = QWidget()
        llm_layout = QFormLayout(self.llm_tab)
        self.llm_provider_combo = QComboBox()
        self.llm_provider_combo.addItems(["openai", "xai", "ollama"])
        llm_layout.addRow("Provider:", self.llm_provider_combo)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        llm_layout.addRow("API Key:", self.api_key_edit)
        self.tabs.addTab(self.llm_tab, "LLM Provider")

        # Buttons ----------------------------------------------------------
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self.load_settings()
        self.populate_databases()
        self._provider_changed(self.llm_provider_combo.currentText())
        self.llm_provider_combo.currentTextChanged.connect(self._provider_changed)

    # ------------------------------------------------------------------
    def populate_databases(self) -> None:
        """List available SQLite databases in the project directory."""
        self.available_list.clear()
        for path in Path.cwd().rglob("*.db"):
            self.available_list.addItem(str(path))

    def _provider_changed(self, provider: str) -> None:
        self.api_key_edit.setEnabled(provider in {"openai", "xai"})

    def load_settings(self) -> None:
        settings = QSettings("LegalAI", "Desktop")
        self.db_type_combo.setCurrentText(settings.value("db/type", "PostgreSQL"))
        self.host_edit.setText(settings.value("db/host", "localhost"))
        self.port_edit.setText(settings.value("db/port", "5432"))
        self.user_edit.setText(settings.value("db/user", ""))
        self.password_edit.setText(settings.value("db/password", ""))
        self.db_name_edit.setText(settings.value("db/name", "legal_ai"))
        provider = settings.value("llm/provider", "openai")
        self.llm_provider_combo.setCurrentText(provider)
        self.api_key_edit.setText(settings.value(f"llm/{provider}_api_key", ""))

    def save_settings(self) -> None:
        settings = QSettings("LegalAI", "Desktop")
        settings.setValue("db/type", self.db_type_combo.currentText())
        settings.setValue("db/host", self.host_edit.text())
        settings.setValue("db/port", self.port_edit.text())
        settings.setValue("db/user", self.user_edit.text())
        settings.setValue("db/password", self.password_edit.text())
        settings.setValue("db/name", self.db_name_edit.text())
        provider = self.llm_provider_combo.currentText()
        settings.setValue("llm/provider", provider)
        settings.setValue(f"llm/{provider}_api_key", self.api_key_edit.text())

    def accept(self) -> None:  # type: ignore[override]
        self.save_settings()
        if self.db_type_combo.currentText() == "PostgreSQL":
            os.environ["POSTGRES_HOST"] = self.host_edit.text()
            os.environ["POSTGRES_PORT"] = self.port_edit.text()
            os.environ["POSTGRES_USER"] = self.user_edit.text()
            os.environ["POSTGRES_PASSWORD"] = self.password_edit.text()
            os.environ["POSTGRES_DB"] = self.db_name_edit.text()
            os.environ["DATABASE_URL"] = (
                f"postgresql://{self.user_edit.text()}:{self.password_edit.text()}@"
                f"{self.host_edit.text()}:{self.port_edit.text()}/{self.db_name_edit.text()}"
            )
        provider = self.llm_provider_combo.currentText()
        os.environ["LLM_PROVIDER"] = provider
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = self.api_key_edit.text()
        elif provider == "xai":
            os.environ["XAI_API_KEY"] = self.api_key_edit.text()
        super().accept()


__all__ = ["StartupConfigDialog"]
