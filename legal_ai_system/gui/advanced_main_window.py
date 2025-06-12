from __future__ import annotations

"""Advanced PyQt6 main window with settings and about dialogs."""

from typing import Any, Dict

from PyQt6 import QtCore, QtGui, QtWidgets

from .main_gui import APIClient, DashboardTab, UploadTab, ReviewTab, WorkflowTab, MonitoringTab, MemoryBrainWidget, StatusTab
from ..core.configuration_manager import ConfigurationManager


class SettingsDialog(QtWidgets.QDialog):
    """Dialog for editing API and authentication settings."""

    settings_saved = QtCore.pyqtSignal()

    def __init__(self, config: ConfigurationManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Settings")

        self.api_edit = QtWidgets.QLineEdit(self.config.get("api_base_url"))
        self.openai_edit = QtWidgets.QLineEdit(self.config.get("openai_api_key", "") or "")
        self.xai_edit = QtWidgets.QLineEdit(self.config.get("xai_api_key", "") or "")

        form = QtWidgets.QFormLayout()
        form.addRow("API Base URL", self.api_edit)
        form.addRow("OpenAI API Key", self.openai_edit)
        form.addRow("xAI API Key", self.xai_edit)

        save_btn = QtWidgets.QPushButton("Save")
        save_btn.clicked.connect(self.save)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(save_btn)

    def save(self) -> None:
        self.config.update_setting("api_base_url", self.api_edit.text().rstrip("/"))
        self.config.update_setting("openai_api_key", self.openai_edit.text() or None)
        self.config.update_setting("xai_api_key", self.xai_edit.text() or None)
        self.settings_saved.emit()
        self.accept()


class AboutDialog(QtWidgets.QDialog):
    """Dialog displaying application information."""

    def __init__(self, config: ConfigurationManager, api: APIClient, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About")

        version = config.get("version", "unknown")
        health: Dict[str, Any] = api.health()
        services = ", ".join(health.get("services_status", {}).keys()) or "N/A"

        label = QtWidgets.QLabel(
            f"<b>Legal AI System</b><br>Version: {version}<br>Active services: {services}"
        )
        label.setTextFormat(QtCore.Qt.TextFormat.RichText)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(label)
        layout.addWidget(close_btn)


class AdvancedMainWindow(QtWidgets.QMainWindow):
    """Main application window with a menu for settings and about."""

    def __init__(self, config: ConfigurationManager) -> None:
        super().__init__()
        self.config = config
        self.setWindowTitle("Legal AI System")
        self.resize(900, 600)

        self.api = APIClient(self.config.get("api_base_url"))
        self.ws_base = self.api.base_url

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(DashboardTab(self.api), "Dashboard")
        tabs.addTab(UploadTab(self.api, self.ws_base), "Document Upload")
        tabs.addTab(ReviewTab(self.api), "Review Queue")
        tabs.addTab(WorkflowTab(self.api), "Workflow Designer")
        tabs.addTab(MonitoringTab(self.api), "Process Monitoring")
        tabs.addTab(MemoryBrainWidget(), "Memory Brain")
        tabs.addTab(StatusTab(self.api), "System Status")

        self.setCentralWidget(tabs)
        self._create_menus()
        self.apply_style()

    def _create_menus(self) -> None:
        bar = self.menuBar()

        file_menu = bar.addMenu("&File")
        settings_action = QtGui.QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings_dialog)
        file_menu.addAction(settings_action)

        help_menu = bar.addMenu("&Help")
        about_action = QtGui.QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def show_settings_dialog(self) -> None:
        dlg = SettingsDialog(self.config, self)
        dlg.settings_saved.connect(self.reload_settings)
        dlg.exec()

    def show_about_dialog(self) -> None:
        AboutDialog(self.config, self.api, self).exec()

    def reload_settings(self) -> None:
        self.api.base_url = self.config.get("api_base_url").rstrip("/")
        self.ws_base = self.api.base_url

    def apply_style(self) -> None:
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor("#1e1e1e"))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#f0f0f0"))
        self.setPalette(palette)
        self.setStyleSheet(
            """
            QWidget { background-color: #2b2b2b; color: #f0f0f0; }
            QPushButton { background-color: #b00020; color: white; border: none; padding: 6px; }
            QLineEdit, QTextEdit { background-color: #3c3c3c; color: #f0f0f0; }
            QTabWidget::pane { border-top: 2px solid #b00020; }
            """
        )


def main() -> None:
    from ..log_setup import init_logging

    init_logging()
    config = ConfigurationManager()
    app = QtWidgets.QApplication([])
    window = AdvancedMainWindow(config)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
