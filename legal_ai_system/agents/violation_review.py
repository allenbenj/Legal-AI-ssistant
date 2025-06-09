# legal_ai_system/agents/violation_review.py
"""SQLite-based Violation Review management.

Provides simple data structures and CRUD utilities used by the
Violation Review GUI. This module stores violations detected by the
system and tracks review status.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ViolationReviewEntry:
    """Represents a legal violation under review."""

    id: str
    case_id: str
    actor: str
    violation_type: str
    statute: Optional[str] = None
    description: str = ""
    suggested_motion: Optional[str] = None
    status: str = "OPEN"  # OPEN, RESOLVED, ESCALATED
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class ViolationReviewManager:
    """CRUD interface for :class:`ViolationReviewEntry` objects."""

    def __init__(self, db_path: str = "violation_review.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # Internal helpers -------------------------------------------------
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    statute TEXT,
                    description TEXT,
                    suggested_motion TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_violations_case ON violations(case_id)"
            )
            conn.commit()

    # CRUD operations --------------------------------------------------
    def insert_violation(self, entry: ViolationReviewEntry) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO violations (
                    id, case_id, actor, violation_type, statute,
                    description, suggested_motion, status,
                    created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.case_id,
                    entry.actor,
                    entry.violation_type,
                    entry.statute,
                    entry.description,
                    entry.suggested_motion,
                    entry.status,
                    entry.created_at,
                    entry.updated_at,
                    json.dumps(entry.metadata) if entry.metadata else None,
                ),
            )
            conn.commit()

    def fetch_violations(
        self,
        case_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ViolationReviewEntry]:
        query = "SELECT * FROM violations WHERE 1=1"
        params: List[Any] = []
        if case_id:
            query += " AND case_id = ?"
            params.append(case_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        results = []
        for row in rows:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            entry = ViolationReviewEntry(
                id=row["id"],
                case_id=row["case_id"],
                actor=row["actor"],
                violation_type=row["violation_type"],
                statute=row["statute"],
                description=row["description"],
                suggested_motion=row["suggested_motion"],
                status=row["status"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=metadata,
            )
            results.append(entry)
        return results

    def fetch_violation(self, violation_id: str) -> Optional[ViolationReviewEntry]:
        """Return a single violation by its ID or ``None`` if not found."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM violations WHERE id = ?",
                (violation_id,),
            ).fetchone()

        if row is None:
            return None

        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        return ViolationReviewEntry(
            id=row["id"],
            case_id=row["case_id"],
            actor=row["actor"],
            violation_type=row["violation_type"],
            statute=row["statute"],
            description=row["description"],
            suggested_motion=row["suggested_motion"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=metadata,
        )

    def update_violation_status(self, violation_id: str, new_status: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute(
                "UPDATE violations SET status = ?, updated_at = ? WHERE id = ?",
                (new_status, datetime.utcnow().isoformat(), violation_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def delete_violation(self, violation_id: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM violations WHERE id = ?",
                (violation_id,),
            )
            conn.commit()
            return cur.rowcount > 0
