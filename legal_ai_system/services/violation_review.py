"""Violation Review Service
==========================

Provides a light weight database backed store for violation records so the
Violation Review GUI can load and update data.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class ViolationReviewEntry:
    """Represents a single violation detected for a document."""

    id: str
    document_id: str
    violation_type: str
    severity: str
    status: str
    description: str
    confidence: float
    detected_time: datetime
    reviewed_by: Optional[str] = None
    review_time: Optional[datetime] = None
    recommended_motion: Optional[str] = None


class ViolationReviewDB:
    """Simple SQLite backed storage for violations."""

    def __init__(self, db_path: str = "./storage/databases/violations.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    detected_time TEXT NOT NULL,
                    reviewed_by TEXT,
                    review_time TEXT,
                    recommended_motion TEXT
                )
                """
            )
            conn.commit()

    def insert_violation(self, entry: ViolationReviewEntry) -> bool:
        """Insert a new violation into the database."""
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO violations (
                        id, document_id, violation_type, severity, status,
                        description, confidence, detected_time, reviewed_by,
                        review_time, recommended_motion
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.id,
                        entry.document_id,
                        entry.violation_type,
                        entry.severity,
                        entry.status,
                        entry.description,
                        entry.confidence,
                        entry.detected_time.isoformat(),
                        entry.reviewed_by,
                        entry.review_time.isoformat() if entry.review_time else None,
                        entry.recommended_motion,
                    ),
                )
                conn.commit()
                return True
        except Exception:
            return False

    def fetch_violations(self, document_id: Optional[str] = None) -> List[ViolationReviewEntry]:
        """Return violations optionally filtered by document id."""
        query = "SELECT * FROM violations"
        params: List[str] = []
        if document_id:
            query += " WHERE document_id = ?"
            params.append(document_id)

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            result: List[ViolationReviewEntry] = []
            for row in rows:
                result.append(
                    ViolationReviewEntry(
                        id=row["id"],
                        document_id=row["document_id"],
                        violation_type=row["violation_type"],
                        severity=row["severity"],
                        status=row["status"],
                        description=row["description"],
                        confidence=row["confidence"],
                        detected_time=datetime.fromisoformat(row["detected_time"]),
                        reviewed_by=row["reviewed_by"],
                        review_time=datetime.fromisoformat(row["review_time"]) if row["review_time"] else None,
                        recommended_motion=row["recommended_motion"],
                    )
                )
            return result

    def update_violation_status(
        self, violation_id: str, status: str, reviewed_by: Optional[str] = None
    ) -> bool:
        """Update the status and review metadata for a violation."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE violations SET status = ?, reviewed_by = ?, review_time = ? WHERE id = ?",
                (status, reviewed_by, datetime.now().isoformat(), violation_id),
            )
            conn.commit()
            return conn.total_changes > 0
