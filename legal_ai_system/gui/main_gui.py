"""Comprehensive PyQt6 GUI for interacting with the Legal AI backend."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PyQt6 import QtCore, QtGui, QtWidgets

from .memory_brain_widget import MemoryBrainWidget

from ..core.settings import settings
from ..log_setup import init_logging


class APIClient:
    """Simple REST client for the backend API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health(self) -> Dict[str, Any]:
        try:
            r = self.session.get(f"{self.base_url}/api/v1/system/health", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # pragma: no cover - network failures
            return {"status": "error", "message": str(exc)}

    def upload(self, file_path: Path) -> Optional[str]:
        try:
            with file_path.open("rb") as fh:
                files = {"file": (file_path.name, fh, "application/octet-stream")}
                r = self.session.post(f"{self.base_url}/api/v1/documents/upload", files=files)
            r.raise_for_status()
            return r.json().get("document_id")
        except Exception:  # pragma: no cover - network failures
            return None

    def process(self, document_id: str, options: Optional[Dict[str, Any]] = None) -> bool:
        try:
            r = self.session.post(
                f"{self.base_url}/api/v1/documents/{document_id}/process",
                json={"processing_options": options or {}},
            )
            r.raise_for_status()
            return True
        except Exception:  # pragma: no cover - network failures
            return False

    def document_status(self, document_id: str) -> Dict[str, Any]:
        try:
            r = self.session.get(f"{self.base_url}/api/v1/documents/{document_id}/status")
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # pragma: no cover - network failures
            return {"status": "error", "message": str(exc)}

    def workflows(self) -> List[Dict[str, Any]]:
        try:
            r = self.session.get(f"{self.base_url}/api/v1/workflows")
            r.raise_for_status()
            return r.json()
        except Exception:  # pragma: no cover - network failures
            return []

    def create_workflow(self, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            r = self.session.post(f"{self.base_url}/api/v1/workflows", json=cfg)
            r.raise_for_status()
            return r.json()
        except Exception:  # pragma: no cover - network failures
            return None

    def update_workflow(self, workflow_id: str, update: Dict[str, Any]) -> bool:
        try:
            r = self.session.put(f"{self.base_url}/api/v1/workflows/{workflow_id}", json=update)
            r.raise_for_status()
            return True
        except Exception:  # pragma: no cover - network failures
            return False

    def delete_workflow(self, workflow_id: str) -> bool:
        try:
            r = self.session.delete(f"{self.base_url}/api/v1/workflows/{workflow_id}")
            return r.status_code in (200, 204)
        except Exception:  # pragma: no cover - network failures
            return False


class WebSocketWorker(QtCore.QThread):
    """Background thread to receive WebSocket messages."""

    message_received = QtCore.pyqtSignal(dict)

    def __init__(self, url: str, topics: List[str]) -> None:
        super().__init__()
        self.url = url
        self.topics = topics
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
            self.message_received.emit(data)

        def on_open(ws: Any) -> None:  # pragma: no cover - GUI thread
            for t in self.topics:
                ws.send(json.dumps({"type": "subscribe", "topic": t}))

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


class DashboardTab(QtWidgets.QWidget):
    """Simple dashboard showing system stats."""

    def __init__(self, api: APIClient) -> None:
        super().__init__()
        self.api = api
        self.info = QtWidgets.QTextEdit(readOnly=True)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.load)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addWidget(refresh)
        self.load()

    def load(self) -> None:
        data = self.api.health()
        self.info.setPlainText(json.dumps(data, indent=2))


class UploadTab(QtWidgets.QWidget):
    """Document upload and processing tab."""

    def __init__(self, api: APIClient, ws_base: str) -> None:
        super().__init__()
        self.api = api
        self.ws_base = ws_base.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self.document_id: Optional[str] = None
        self.ws_worker: Optional[WebSocketWorker] = None

        self.file_edit = QtWidgets.QLineEdit(readOnly=True)
        browse_btn = QtWidgets.QPushButton("Browse")
        browse_btn.clicked.connect(self.choose_file)
        upload_btn = QtWidgets.QPushButton("Upload && Process")
        upload_btn.clicked.connect(self.upload)
        self.progress = QtWidgets.QProgressBar()
        self.progress_anim = QtCore.QPropertyAnimation(self.progress, b"value", self)
        self.progress_anim.setDuration(400)
        self.progress_anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self.output = QtWidgets.QTextEdit(readOnly=True)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.file_edit)
        top.addWidget(browse_btn)
        top.addWidget(upload_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.progress)
        layout.addWidget(self.output)

    def choose_file(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Document")
        if file_path:
            self.file_edit.setText(file_path)

    def upload(self) -> None:
        path_str = self.file_edit.text()
        if not path_str:
            return
        path = Path(path_str)
        doc_id = self.api.upload(path)
        if not doc_id:
            QtWidgets.QMessageBox.warning(self, "Upload Failed", "Unable to upload file")
            return
        self.document_id = doc_id
        success = self.api.process(doc_id)
        if not success:
            QtWidgets.QMessageBox.warning(self, "Process Failed", "Unable to start processing")
            return
        self.progress.setValue(0)
        self.output.append(f"Processing {doc_id} ...")
        self.start_ws()

    def start_ws(self) -> None:
        if not self.document_id:
            return
        url = f"{self.ws_base}/ws/{self.client_id}"
        topics = [f"document_updates_{self.document_id}"]
        self.ws_worker = WebSocketWorker(url, topics)
        self.ws_worker.message_received.connect(self.handle_update)
        self.ws_worker.start()

    def handle_update(self, data: Dict[str, Any]) -> None:
        if data.get("document_id") != self.document_id:
            return
        if data.get("type") == "processing_progress":
            prog = int(float(data.get("progress", 0)) * 100)
            self.progress_anim.stop()
            self.progress_anim.setStartValue(self.progress.value())
            self.progress_anim.setEndValue(prog)
            self.progress_anim.start()
            self.output.append(f"Stage: {data.get('stage')}")
        elif data.get("type") == "processing_complete":
            self.progress_anim.stop()
            self.progress_anim.setStartValue(self.progress.value())
            self.progress_anim.setEndValue(100)
            self.progress_anim.start()
            self.output.append("Completed")


class ReviewTab(QtWidgets.QWidget):
    """Basic review queue interaction."""

    def __init__(self, api: APIClient) -> None:
        super().__init__()
        self.api = api
        self.queue_info = QtWidgets.QLabel()
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.load)

        self.item_edit = QtWidgets.QLineEdit()
        self.decision_box = QtWidgets.QComboBox()
        self.decision_box.addItems(["approve", "reject", "modify"])
        self.notes_edit = QtWidgets.QTextEdit()
        submit_btn = QtWidgets.QPushButton("Submit Decision")
        submit_btn.clicked.connect(self.submit)

        form = QtWidgets.QFormLayout()
        form.addRow("Item ID", self.item_edit)
        form.addRow("Decision", self.decision_box)
        form.addRow("Notes", self.notes_edit)
        form.addRow(submit_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.queue_info)
        layout.addWidget(refresh)
        layout.addLayout(form)
        self.load()

    def load(self) -> None:
        data = self.api.health()
        cnt = data.get("pending_reviews_count", 0)
        self.queue_info.setText(f"Pending reviews: {cnt}")

    def submit(self) -> None:
        item_id = self.item_edit.text().strip()
        if not item_id:
            return
        payload = {
            "item_id": item_id,
            "decision": self.decision_box.currentText(),
            "reviewer_notes": self.notes_edit.toPlainText(),
        }
        try:
            r = self.api.session.post(f"{self.api.base_url}/api/v1/calibration/review", json=payload)
            r.raise_for_status()
            QtWidgets.QMessageBox.information(self, "Success", "Decision submitted")
        except Exception as exc:  # pragma: no cover - network failures
            QtWidgets.QMessageBox.warning(self, "Error", str(exc))


class WorkflowTab(QtWidgets.QWidget):
    """Workflow designer tab."""

    def __init__(self, api: APIClient) -> None:
        super().__init__()
        self.api = api
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["ID", "Name"])
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.load)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addWidget(refresh)
        self.load()

    def load(self) -> None:
        workflows = self.api.workflows()
        self.table.setRowCount(len(workflows))
        for row, wf in enumerate(workflows):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(wf.get("id", "")))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(wf.get("name", "")))


class MonitoringTab(QtWidgets.QWidget):
    """Shows active document processing."""

    def __init__(self, api: APIClient) -> None:
        super().__init__()
        self.api = api
        self.output = QtWidgets.QTextEdit(readOnly=True)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(5000)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.output)
        self.refresh()

    def refresh(self) -> None:
        data = self.api.health()
        active = data.get("active_documents_count", 0)
        self.output.setPlainText(f"Active documents: {active}\n{json.dumps(data, indent=2)}")


class StatusTab(QtWidgets.QWidget):
    """Displays overall system status."""

    def __init__(self, api: APIClient) -> None:
        super().__init__()
        self.api = api
        self.text = QtWidgets.QTextEdit(readOnly=True)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.load)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.text)
        layout.addWidget(refresh)
        self.load()

    def load(self) -> None:
        data = self.api.health()
        self.text.setPlainText(json.dumps(data, indent=2))


class SettingsTab(QtWidgets.QWidget):
    """Application settings tab."""

    settings_changed = QtCore.pyqtSignal()

    def __init__(self, api_client: APIClient) -> None:
        super().__init__()
        self.api_client = api_client

        self.api_edit = QtWidgets.QLineEdit(api_client.base_url)
        self.openai_key = QtWidgets.QLineEdit(settings.openai_api_key or "")
        self.xai_key = QtWidgets.QLineEdit(settings.xai_api_key or "")

        save_btn = QtWidgets.QPushButton("Save")
        save_btn.clicked.connect(self.save)

        form = QtWidgets.QFormLayout(self)
        form.addRow("API Base URL", self.api_edit)
        form.addRow("OpenAI API Key", self.openai_key)
        form.addRow("XAI API Key", self.xai_key)
        form.addRow(save_btn)

    def save(self) -> None:
        self.api_client.base_url = self.api_edit.text().rstrip("/")
        settings.openai_api_key = self.openai_key.text() or None
        settings.xai_api_key = self.xai_key.text() or None
        self.settings_changed.emit()


class MainWindow(QtWidgets.QMainWindow):
    """Main application window with tabbed interface."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Legal AI System")
        self.resize(900, 600)

        self.api = APIClient(settings.api_base_url)
        self.ws_base = settings.api_base_url

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(DashboardTab(self.api), "Dashboard")
        tabs.addTab(UploadTab(self.api, self.ws_base), "Document Upload")
        tabs.addTab(ReviewTab(self.api), "Review Queue")
        tabs.addTab(WorkflowTab(self.api), "Workflow Designer")
        tabs.addTab(MonitoringTab(self.api), "Process Monitoring")
        tabs.addTab(MemoryBrainWidget(), "Memory Brain")
        tabs.addTab(StatusTab(self.api), "System Status")
        settings_tab = SettingsTab(self.api)
        settings_tab.settings_changed.connect(self.reload_settings)
        tabs.addTab(settings_tab, "Settings")

        self.setCentralWidget(tabs)
        self.apply_style()

    def reload_settings(self) -> None:
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
    init_logging()
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()


__all__ = ["main", "MainWindow"]

