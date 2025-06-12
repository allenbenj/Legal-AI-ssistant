"""Networking stubs for the integrated GUI."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

from PyQt6.QtCore import QObject, QThread, pyqtSignal


class NetworkManager(QObject):
    """Basic network manager holding a base URL."""

    connectionStatusChanged = pyqtSignal(str)

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.connectionStatusChanged.emit("connected")


class LegalAIAPIClient(QObject):
    """Simplified API client emitting Qt signals."""

    documentsLoaded = pyqtSignal(list)
    documentUploaded = pyqtSignal(str)
    processingComplete = pyqtSignal(str, dict)

    def __init__(self, _network: NetworkManager) -> None:
        super().__init__()

    # Methods used by the GUI -------------------------------------------------
    def loadDocuments(self) -> None:
        self.documentsLoaded.emit([])

    def uploadDocument(self, path: Path, options: Dict[str, Any]) -> None:
        self.documentUploaded.emit(path.stem)


class DocumentProcessingWorker(QThread):  # pragma: no cover - thread logic
    """Background worker that drives the WorkflowOrchestrator."""

    progress = pyqtSignal(str, int, str)

    def __init__(
        self, _client: LegalAIAPIClient, service_container: Any | None = None
    ) -> None:
        super().__init__()
        self._queue: List[tuple[str, Path, Dict[str, Any]]] = []
        self.service_container = service_container

    def addDocument(self, doc_id: str, path: Path, options: Dict[str, Any]) -> None:
        self._queue.append((doc_id, path, options))
        if not self.isRunning():
            self.start()

    def run(self) -> None:
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        orchestrator = None
        if self.service_container is not None:
            try:
                orchestrator = loop.run_until_complete(
                    self.service_container.get_workflow_orchestrator()
                )
            except Exception:
                orchestrator = None

        for doc_id, path, opts in self._queue:
            if orchestrator is not None:
                orchestrator.workflow.register_progress_callback(
                    lambda msg, prog, did=doc_id: self.progress.emit(
                        did, int(prog * 100), msg
                    )
                )
                loop.run_until_complete(
                    orchestrator.execute_workflow_instance(
                        document_path_str=str(path),
                        custom_metadata={"document_id": doc_id, **opts},
                    )
                )
            else:
                for pct in range(0, 101, 20):
                    self.progress.emit(doc_id, pct, "processing")

        self._queue.clear()


class WebSocketClient(QObject):  # pragma: no cover - stub
    """Very small WebSocket client placeholder."""

    connected = pyqtSignal()
    messageReceived = pyqtSignal(dict)

    def __init__(self, _url: str) -> None:
        super().__init__()

    def connect(self) -> None:
        self.connected.emit()

    def disconnect(self) -> None:
        pass


class AsyncAPIClient:  # pragma: no cover - unused helper
    pass


__all__ = [
    "NetworkManager",
    "LegalAIAPIClient",
    "DocumentProcessingWorker",
    "WebSocketClient",
    "AsyncAPIClient",
]
