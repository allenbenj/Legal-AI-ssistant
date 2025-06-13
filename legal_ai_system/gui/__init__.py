"""GUI components for the Legal AI System."""

def __getattr__(name):
    if name == "IntegratedMainWindow":
        from .legal_ai_pyqt6_enhanced import IntegratedMainWindow
        return IntegratedMainWindow
    if name == "main":
        from .legal_ai_pyqt6_enhanced import main
        return main
    if name == "StartupConfigDialog":
        from .startup_config_dialog import StartupConfigDialog
        return StartupConfigDialog
    if name in {"DraggableComponentButton", "WorkflowCanvas", "WorkflowBuilderWidget"}:
        from .workflow_builder import (
            DraggableComponentButton,
            WorkflowCanvas,
            WorkflowBuilderWidget,
        )
        return locals()[name]
    raise AttributeError(name)

__all__ = [
    "IntegratedMainWindow",
    "main",
    "StartupConfigDialog",
    "DraggableComponentButton",
    "WorkflowCanvas",
    "WorkflowBuilderWidget",
]

