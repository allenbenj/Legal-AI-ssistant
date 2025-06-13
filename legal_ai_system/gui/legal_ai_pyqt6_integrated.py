from __future__ import annotations

"""Integrated PyQt6 GUI for the Legal AI System."""

from pathlib import Path
from typing import Iterable, Optional

from PyQt6.QtWidgets import QFileDialog

from .backend_bridge import BackendBridge
from .legal_ai_pyqt6_enhanced import EnhancedMainWindow


class IntegratedMainWindow(EnhancedMainWindow):
    """Main window wired to the asynchronous backend via :class:`BackendBridge`."""

    def __init__(self, backend_bridge: Optional[BackendBridge] = None) -> None:
        self.backend_bridge = backend_bridge or BackendBridge()
        super().__init__()
        try:
            self.backend_bridge.start()
        except Exception as exc:  # pragma: no cover - bridge start failures
            self.log(f"Backend bridge start failed: {exc}")
        if hasattr(self.backend_bridge, "serviceReady"):
            self.backend_bridge.serviceReady.connect(self.on_backend_ready)

    # ------------------------------------------------------------------
    # Backend interaction helpers
    # ------------------------------------------------------------------
    def on_backend_ready(self) -> None:
        self.log("Backend services initialised")

    def _submit_files(self, files: Iterable[str]) -> None:
        """Send selected files to the backend for processing."""
        for file_path in files:
            try:
                self.backend_bridge.upload_document(Path(file_path), {})
            except Exception as exc:  # pragma: no cover - upload failures
                self.log(f"Failed to submit {file_path}: {exc}")

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def upload_documents(self) -> None:  # pragma: no cover - GUI action
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Documents",
            "",
            "Documents (*.pdf *.docx *.txt *.md *.markdown *.json *.csv);;All Files (*)",
        )
        if not files:
            self.log("No files selected")
            return
        self._submit_files(files)


# Preserve previous API -------------------------------------------------
from .legal_ai_pyqt6_enhanced import main as main
__all__ = ["IntegratedMainWindow", "main"]
