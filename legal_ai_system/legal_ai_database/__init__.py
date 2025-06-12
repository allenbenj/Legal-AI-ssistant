# AGENT_STUB
"""Simplified database utilities and preference storage."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QObject, pyqtSignal


class DatabaseManager(QObject):
    """Very small database manager stub."""

    databaseReady = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, db_path: str | Path = "legal_ai_gui.db") -> None:
        super().__init__()
        self.db_path = Path(db_path)
        self.databaseReady.emit()

    # Minimal API used by the GUI
    def saveDocument(self, doc_id: str, filename: str, file_size: int = 0, metadata: Optional[Dict] = None) -> None:
        pass

    def updateDocumentStatus(self, doc_id: str, status: str, results: Optional[Dict] = None) -> None:
        pass

    def getDocuments(self, limit: int = 100) -> List[Dict[str, Any]]:
        return []


class CacheManager:
    """No-op cache manager."""

    def __init__(self, _db: DatabaseManager | None = None) -> None:
        self._cache: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value


class PreferencesManager(QObject):
    """In-memory preferences with change notification."""

    preferenceChanged = pyqtSignal(str, object)

    def __init__(self, _db: DatabaseManager | None = None) -> None:
        super().__init__()
        self._prefs: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._prefs.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._prefs[key] = value
        self.preferenceChanged.emit(key, value)


class DocumentSearchEngine:
    """Naive in-memory search engine."""

    def __init__(self, _db: DatabaseManager | None = None) -> None:
        self._index: Dict[str, Dict[str, Any]] = {}

    def indexDocument(self, doc_id: str, filename: str, text: str, metadata: Dict[str, Any]) -> None:
        self._index[doc_id] = {"filename": filename, "text": text, "metadata": metadata}

    def search(self, query: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for doc_id, data in self._index.items():
            if query.lower() in data.get("text", "").lower():
                snippet = data.get("text", "")[:100]
                results.append({"document_id": doc_id, "filename": data["filename"], "snippet": snippet})
        return results


__all__ = [
    "DatabaseManager",
    "CacheManager",
    "PreferencesManager",
    "DocumentSearchEngine",
]
