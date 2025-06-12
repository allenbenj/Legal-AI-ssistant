"""PyQt GUI package for the Legal AI System."""

from .main_gui import MainWindow
from .windows import (
    DocumentViewerWindow,
    WorkflowDesignerWindow,
    ReviewQueueWindow,
    DocumentTableModel,
    ViolationsTableModel,
)

__all__ = [
    "MainWindow",
    "DocumentViewerWindow",
    "WorkflowDesignerWindow",
    "ReviewQueueWindow",
    "DocumentTableModel",
    "ViolationsTableModel",
]
