"""Local persistence layer for the PyQt GUI."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QObject, pyqtSignal


class DatabaseManager(QObject):
    """Simple SQLite based document store."""

    databaseReady = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, db_path: str | Path = "legal_ai_gui.db") -> None:
        super().__init__()
        self.db_path = Path(db_path)
        try:
            self._conn = sqlite3.connect(self.db_path)
            self._setup()
            self.databaseReady.emit()
        except Exception as e:  # pragma: no cover - sqlite failure
            self.error.emit(str(e))

    def _setup(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_size INTEGER,
                status TEXT,
                metadata TEXT
            )
            """
        )
        self._conn.commit()

    def saveDocument(
        self, doc_id: str, filename: str, file_size: int = 0, metadata: Optional[Dict] = None
    ) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO documents (id, filename, file_size, status, metadata) VALUES (?, ?, ?, ?, ?)",
            (
                doc_id,
                filename,
                file_size,
                "pending",
                json.dumps(metadata or {}),
            ),
        )
        self._conn.commit()

    def updateDocumentStatus(
        self, doc_id: str, status: str, results: Optional[Dict] = None
    ) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE documents SET status = ?, metadata = ? WHERE id = ?",
            (status, json.dumps(results or {}), doc_id),
        )
        self._conn.commit()

    def getDocuments(self, limit: int = 100) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT id, filename, file_size, status, metadata FROM documents LIMIT ?", (limit,))
        rows = cur.fetchall()
        docs = []
        for row in rows:
            meta = json.loads(row[4]) if row[4] else {}
            docs.append(
                {
                    "document_id": row[0],
                    "filename": row[1],
                    "file_size": row[2],
                    "status": row[3],
                    "metadata": meta,
                }
            )
        return docs


class CacheManager:
    """Very small in-memory cache."""

    def __init__(self, _db: DatabaseManager | None = None) -> None:
        self._cache: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value


class PreferencesManager(QObject):
    """Preferences stored in memory and persisted via :class:`DatabaseManager`."""

    preferenceChanged = pyqtSignal(str, object)

    def __init__(self, db: DatabaseManager | None = None) -> None:
        super().__init__()
        self._prefs: Dict[str, Any] = {}
        self._db = db

    def get(self, key: str, default: Any = None) -> Any:
        return self._prefs.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._prefs[key] = value
        self.preferenceChanged.emit(key, value)


class DocumentSearchEngine:
    """Naive in-memory search helper."""

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
