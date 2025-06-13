"""Asynchronous networking layer used by the PyQt GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from legal_ai_system.gui.backend_bridge import BackendBridge


class NetworkManager(QObject):
    """Provides connection status information for the GUI."""

    connectionStatusChanged = pyqtSignal(str)

    def __init__(self, bridge: BackendBridge) -> None:
        super().__init__()
        self.bridge = bridge
        self.connectionStatusChanged.emit("connected")

    def checkConnection(self) -> None:
        status = "connected" if self.bridge.is_ready() else "disconnected"
        self.connectionStatusChanged.emit(status)


class LegalAIAPIClient(QObject):
    """Qt-friendly API client forwarding requests to the backend bridge."""

    documentsLoaded = pyqtSignal(list)
    documentUploaded = pyqtSignal(str)
    processingProgress = pyqtSignal(str, int, str)

    def __init__(self, bridge: BackendBridge) -> None:
        super().__init__()
        self.bridge = bridge
        self.bridge.processingProgress.connect(self.processingProgress.emit)

    def loadDocuments(self) -> None:
        future = self.bridge.get_status("all")  # placeholder for future list API
        future.add_done_callback(lambda f: self.documentsLoaded.emit(f.result() or []))

    def uploadDocument(self, path: Path, options: Dict[str, Any]) -> None:
        future = self.bridge.upload_document(path, options)

        def _done(f: Any) -> None:
            try:
                res = f.result()
                self.documentUploaded.emit(res.get("document_id", path.stem))
            except Exception:
                self.documentUploaded.emit(path.stem)

        future.add_done_callback(_done)


class DocumentProcessingWorker(QThread):
    """Processes queued documents using the API client."""

    progress = pyqtSignal(str, int, str)

    def __init__(self, client: LegalAIAPIClient) -> None:
        super().__init__()
        self.client = client
        self._queue: List[tuple[str, Path, Dict[str, Any]]] = []
        self.client.processingProgress.connect(self.progress.emit)

    def addDocument(self, doc_id: str, path: Path, options: Dict[str, Any]) -> None:
        self._queue.append((doc_id, path, options))
        if not self.isRunning():
            self.start()

    def run(self) -> None:  # pragma: no cover - thread logic
        while self._queue:
            _doc_id, path, opts = self._queue.pop(0)
            future = self.client.bridge.upload_document(path, opts)
            future.result()


class WebSocketClient(QObject):  # pragma: no cover - placeholder
    connected = pyqtSignal()
    messageReceived = pyqtSignal(dict)

    def __init__(self, _url: str) -> None:
        super().__init__()

    def connect(self) -> None:
        self.connected.emit()

    def disconnect(self) -> None:
        pass


class AsyncAPIClient:
    """Lightweight awaitable API client for non-Qt use."""

    def __init__(self, bridge: BackendBridge) -> None:
        self.bridge = bridge

    async def upload_document(self, path: Path, options: Dict[str, Any]) -> Dict[str, Any]:
        return await self.bridge._upload_document_async(path, options)

