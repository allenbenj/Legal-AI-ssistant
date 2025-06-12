"""GUI components for the Legal AI System."""

from .legal_ai_pyqt6_integrated import IntegratedMainWindow, main
from .workflow_builder import (
    DraggableComponentButton,
    WorkflowCanvas,
    WorkflowBuilderWidget,
)

__all__ = [
    "IntegratedMainWindow",
    "main",
    "DraggableComponentButton",
    "WorkflowCanvas",
    "WorkflowBuilderWidget",
]

