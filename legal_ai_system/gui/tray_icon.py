from __future__ import annotations

import json
import threading
import uuid
from typing import Any, Dict

from PyQt6 import QtCore, QtGui, QtWidgets

from ..core.settings import settings


class _StatusWorker(QtCore.QThread):
    """Background WebSocket client for system status updates."""

    status_received = QtCore.pyqtSignal(dict)

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # pragma: no cover - GUI thread
        try:
            import websocket  # type: ignore
        except Exception:
            return

        def on_message(ws: Any, message: str) -> None:  # pragma: no cover - GUI thread
            try:
                data = json.loads(message)
            except Exception:
                return
            if data.get("type") == "system_status":
                self.status_received.emit(data)

        def on_open(ws: Any) -> None:  # pragma: no cover - GUI thread
            ws.send(json.dumps({"type": "subscribe", "topic": "system_status"}))

        ws = websocket.WebSocketApp(
            self.url,
            on_message=on_message,
            on_open=on_open,
        )

        while not self._stop.is_set():  # pragma: no cover - GUI thread
            try:
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                pass
            if not self._stop.wait(2):
                continue


class TrayIcon(QtWidgets.QSystemTrayIcon):
    """System tray icon displaying realtime system status."""

    def __init__(self, main_window: QtWidgets.QMainWindow) -> None:
        icon = main_window.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ComputerIcon)
        super().__init__(icon, main_window)
        self.main_window = main_window

        menu = QtWidgets.QMenu()
        open_action = menu.addAction("Open Main Window")
        open_action.triggered.connect(self.show_main_window)
        analytics_action = menu.addAction("Show Analytics")
        analytics_action.triggered.connect(self.show_analytics)
        exit_action = menu.addAction("Exit")
        exit_action.triggered.connect(QtWidgets.QApplication.quit)
        self.setContextMenu(menu)

        ws_url = f"{settings.api_base_url.rstrip('/')}/ws/{uuid.uuid4().hex}"
        self._worker = _StatusWorker(ws_url)
        self._worker.status_received.connect(self.update_tooltip)
        self._worker.start()
        self.setToolTip("Starting...")

    def show_main_window(self) -> None:
        self.main_window.showNormal()
        self.main_window.raise_()
        self.main_window.activateWindow()

    def show_analytics(self) -> None:
        url = f"{settings.api_base_url.rstrip('/')}/analytics"
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))

    def update_tooltip(self, data: Dict[str, Any]) -> None:
        cpu = data.get("cpu")
        mem = data.get("memory")
        disk = data.get("disk")
        self.setToolTip(f"CPU {cpu}% | Mem {mem}% | Disk {disk}%")

    def shutdown(self) -> None:
        if self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(1000)
        self.hide()
