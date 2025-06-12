from __future__ import annotations

from typing import Dict, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QDialog,
    QFormLayout,
    QLineEdit,
    QDialogButtonBox,
)


class AgentSettingsDialog(QDialog):
    """Simple settings dialog for configuring an agent."""

    def __init__(self, agent_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{agent_name} Settings")
        layout = QFormLayout(self)
        self.endpoint_edit = QLineEdit()
        layout.addRow("API Endpoint", self.endpoint_edit)
        self.token_edit = QLineEdit()
        layout.addRow("Token", self.token_edit)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class AgentCategoryTab(QWidget):
    """Tab widget containing controls and queue for an agent category."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        layout = QVBoxLayout(self)

        control_bar = QWidget()
        control_layout = QHBoxLayout(control_bar)
        self.start_btn = QPushButton("Start All")
        self.stop_btn = QPushButton("Stop All")
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addStretch()
        control_layout.addWidget(self.settings_btn)
        layout.addWidget(control_bar)

        self.queue = QTreeWidget()
        self.queue.setHeaderLabels(["Task", "Status", "Progress"])
        layout.addWidget(self.queue)

        self.monitor = QTimer(self)
        self.monitor.timeout.connect(self.update_progress)
        self.monitor.start(1000)

    def open_settings(self) -> None:  # pragma: no cover - GUI logic
        dlg = AgentSettingsDialog(self.title, self)
        dlg.exec()

    def add_task(self, task_name: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([task_name, "Pending", "0%"])
        self.queue.addTopLevelItem(item)
        return item

    def update_task_progress(self, item: QTreeWidgetItem, progress: int) -> None:
        item.setText(1, "Running" if progress < 100 else "Done")
        item.setText(2, f"{progress}%")

    def update_progress(self) -> None:  # pragma: no cover - placeholder
        # Real monitoring would pull task states from backend
        pass


class AgentManagerWidget(QWidget):
    """Central widget managing all AI agents."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.categories: Dict[str, AgentCategoryTab] = {}
        self._create_default_tabs()

    def _create_default_tabs(self) -> None:
        self.add_category("Document Agents")
        self.add_category("Analysis Agents")
        self.add_category("Utilities")

    def add_category(self, name: str) -> AgentCategoryTab:
        tab = AgentCategoryTab(name)
        self.tabs.addTab(tab, name)
        self.categories[name] = tab
        return tab

    def add_task(self, category: str, task_name: str) -> Optional[QTreeWidgetItem]:
        tab = self.categories.get(category)
        if tab:
            return tab.add_task(task_name)
        return None


