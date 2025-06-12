from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Awaitable
from datetime import datetime, timezone

from PyQt6.QtCore import QObject, pyqtSignal

from legal_ai_system.services.service_container import create_service_container, ServiceContainer
from legal_ai_system.services.integration_service import (
    create_integration_service,
    LegalAIIntegrationService,
)
from legal_ai_system.services.security_manager import User, AccessLevel


class BackendBridge(QObject):
    """Bridge between the PyQt GUI and the asynchronous backend services."""

    serviceReady = pyqtSignal()
    processingProgress = pyqtSignal(str, float, str)

    def __init__(self) -> None:
        super().__init__()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._container: Optional[ServiceContainer] = None
        self._integration_service: Optional[LegalAIIntegrationService] = None
        self._ready = False

    def start(self) -> None:
        """Start the backend event loop and initialise services."""
        self._thread.start()
        self.run_async(self._initialize())

    def is_ready(self) -> bool:
        return self._ready

    def run_async(self, coro: Awaitable[Any]) -> asyncio.Future:
        """Schedule a coroutine on the bridge's loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _initialize(self) -> None:
        self._container = await create_service_container()
        await self._container.register_service(
            "integration_service",
            factory=lambda sc: create_integration_service(sc),
            is_async_factory=False,
        )
        await self._container.initialize_all_services()
        self._integration_service = await self._container.get_service("integration_service")
        self._ready = True
        self.serviceReady.emit()

    def shutdown(self) -> None:
        """Shutdown services and stop the loop."""
        if self._container:
            self.run_async(self._container.shutdown_all_services()).result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    async def _upload_document_async(
        self, file_path: Path, options: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self._integration_service:
            raise RuntimeError("Integration service not ready")

        with open(file_path, "rb") as f:
            content = f.read()

        user = User(
            user_id="gui_user",
            username="gui",
            email="gui@example.com",
            password_hash="",
            salt="",
            access_level=AccessLevel.WRITE,
            created_at=datetime.now(timezone.utc),
        )

        async def cb(message: str, progress: float) -> None:
            self.processingProgress.emit(file_path.name, progress * 100, message)

        return await self._integration_service.upload_and_process_document(
            file_content=content,
            filename=file_path.name,
            user=user,
            options=options,
            progress_cb=cb,
        )

    def upload_document(self, file_path: Path, options: Dict[str, Any]) -> asyncio.Future:
        """Upload a document and start processing."""
        return self.run_async(self._upload_document_async(file_path, options))

    async def _get_status_async(self, document_id: str) -> Dict[str, Any]:
        if not self._integration_service:
            return {}
        user = User(
            user_id="gui_user",
            username="gui",
            email="gui@example.com",
            password_hash="",
            salt="",
            access_level=AccessLevel.READ,
            created_at=datetime.now(timezone.utc),
        )
        return await self._integration_service.get_document_analysis_status(document_id, user)

    def get_status(self, document_id: str) -> asyncio.Future:
        return self.run_async(self._get_status_async(document_id))

