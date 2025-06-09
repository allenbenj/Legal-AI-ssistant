"""Legal Violation Review (LVR) module.

This module manages potential violations of law or ethics detected during
case analysis. Violations are stored in an SQLite database for structured
querying and mirrored to a JSONL file for transparency/audit purposes.

The schema roughly follows the design discussed in planning notes.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

import numpy as np
import faiss  # type: ignore


@dataclass
class LVRActor:
    """Actor involved in a potential violation."""

    name: str
    role: str


@dataclass
class EvidenceLink:
    """Link to supporting evidence."""

    type: str
    description: Optional[str] = None
    path: Optional[str] = None
    page: Optional[int] = None
    ref_id: Optional[str] = None


@dataclass
class ReviewNote:
    """Audit trail note for a violation."""

    reviewed_by: str
    comment: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class LVRRecord:
    """Primary record stored by the Legal Violation Review system."""

    case_id: str
    violation_type: str
    statute: str
    jurisdiction: str
    actor: LVRActor
    related_statements: List[str] = field(default_factory=list)
    evidence_links: List[EvidenceLink] = field(default_factory=list)
    harms: List[str] = field(default_factory=list)
    status: str = "Open"
    created_by: str = ""
    review_notes: List[ReviewNote] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class LegalViolationReview:
    """Manage Legal Violation Review records."""

    def __init__(
        self,
        db_path: str = "./storage/databases/lvr.db",
        jsonl_path: str = "./storage/lvr_records.jsonl",
        faiss_index_path: str = "./storage/faiss/lvr.index",
    ) -> None:
        self.db_path = Path(db_path)
        self.jsonl_path = Path(jsonl_path)
        self.faiss_path = Path(faiss_index_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialized = False
        self.faiss_dim = 32
        self.faiss_index = self._load_faiss_index()
        self._id_map: Dict[int, str] = {}

    def initialize(self) -> None:
        """Initialise SQLite schema."""
        if self._initialized:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    violation_id TEXT PRIMARY KEY,
                    case_id TEXT,
                    violation_type TEXT,
                    statute TEXT,
                    jurisdiction TEXT,
                    actor_name TEXT,
                    actor_role TEXT,
                    related_statements TEXT,
                    evidence_links TEXT,
                    harms TEXT,
                    status TEXT,
                    created_by TEXT,
                    review_notes TEXT,
                    tags TEXT,
                    created_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS violation_memory_links (
                    violation_id TEXT,
                    memory_entry_id TEXT,
                    PRIMARY KEY (violation_id, memory_entry_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS faiss_map (
                    faiss_id INTEGER PRIMARY KEY,
                    violation_id TEXT UNIQUE
                )
                """
            )
            rows = conn.execute("SELECT faiss_id, violation_id FROM faiss_map").fetchall()
            self._id_map = {row[0]: row[1] for row in rows}
            conn.commit()
        self._initialized = True

    def _load_faiss_index(self) -> faiss.IndexIDMap:
        if self.faiss_path.exists():
            return faiss.read_index(str(self.faiss_path))
        return faiss.IndexIDMap(faiss.IndexFlatL2(self.faiss_dim))

    def _save_faiss_index(self) -> None:
        faiss.write_index(self.faiss_index, str(self.faiss_path))

    def _embed_text(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = np.frombuffer(digest[: self.faiss_dim], dtype=np.uint8).astype(
            "float32"
        )
        return vec

    def _embed_record(self, record: LVRRecord) -> np.ndarray:
        text = " ".join(
            [
                record.case_id,
                record.violation_type,
                record.statute,
                record.jurisdiction,
                " ".join(record.harms),
            ]
        )
        return self._embed_text(text)

    def _generate_faiss_id(self, violation_id: str) -> int:
        return abs(hash(violation_id)) % (2**63)

    def search_similar(self, text: str, k: int = 5) -> List[LVRRecord]:
        self.initialize()
        if self.faiss_index.ntotal == 0:
            return []
        query_vec = self._embed_text(text).reshape(1, -1)
        distances, ids = self.faiss_index.search(query_vec, k)
        results: List[LVRRecord] = []
        for idx in ids[0]:
            if idx == -1:
                continue
            vid = self._id_map.get(idx)
            if vid:
                rec = self.get_violation(vid)
                if rec:
                    results.append(rec)
        return results

    def add_violation(self, record: LVRRecord) -> None:
        """Persist a violation record."""
        self.initialize()
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO violations (
                    violation_id,
                    case_id,
                    violation_type,
                    statute,
                    jurisdiction,
                    actor_name,
                    actor_role,
                    related_statements,
                    evidence_links,
                    harms,
                    status,
                    created_by,
                    review_notes,
                    tags,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.violation_id,
                    record.case_id,
                    record.violation_type,
                    record.statute,
                    record.jurisdiction,
                    record.actor.name,
                    record.actor.role,
                    json.dumps(record.related_statements),
                    json.dumps([asdict(e) for e in record.evidence_links]),
                    json.dumps(record.harms),
                    record.status,
                    record.created_by,
                    json.dumps([asdict(n) for n in record.review_notes]),
                    json.dumps(record.tags),
                    record.created_at,
                ),
            )
            conn.commit()

            faiss_id = self._generate_faiss_id(record.violation_id)
            embedding = self._embed_record(record).reshape(1, -1)
            self.faiss_index.add_with_ids(embedding, np.array([faiss_id], dtype="int64"))
            conn.execute(
                "INSERT OR REPLACE INTO faiss_map (faiss_id, violation_id) VALUES (?, ?)",
                (faiss_id, record.violation_id),
            )
            self._id_map[faiss_id] = record.violation_id
            self._save_faiss_index()

        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def get_violation(self, violation_id: str) -> Optional[LVRRecord]:
        """Retrieve a violation record by ID."""
        self.initialize()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM violations WHERE violation_id = ?",
                (violation_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def update_status(self, violation_id: str, status: str) -> bool:
        """Update the status of a violation."""
        self.initialize()
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE violations SET status = ? WHERE violation_id = ?",
                (status, violation_id),
            )
            conn.commit()
            return conn.total_changes > 0

    def add_review_note(self, violation_id: str, note: ReviewNote) -> bool:
        """Append a review note to a violation."""
        self.initialize()
        rec = self.get_violation(violation_id)
        if not rec:
            return False
        rec.review_notes.append(note)
        self.add_violation(rec)
        return True

    def add_evidence(self, violation_id: str, evidence: EvidenceLink) -> bool:
        """Attach additional evidence to a violation."""
        self.initialize()
        rec = self.get_violation(violation_id)
        if not rec:
            return False
        rec.evidence_links.append(evidence)
        self.add_violation(rec)
        return True

    def list_violations(self, case_id: Optional[str] = None) -> List[LVRRecord]:
        """Return all violation records, optionally filtered by case."""
        self.initialize()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if case_id:
                rows = conn.execute(
                    "SELECT * FROM violations WHERE case_id = ? ORDER BY created_at",
                    (case_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM violations ORDER BY created_at"
                ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row: sqlite3.Row) -> LVRRecord:
        return LVRRecord(
            case_id=row["case_id"],
            violation_type=row["violation_type"],
            statute=row["statute"],
            jurisdiction=row["jurisdiction"],
            actor=LVRActor(name=row["actor_name"], role=row["actor_role"]),
            related_statements=json.loads(row["related_statements"]),
            evidence_links=[EvidenceLink(**e) for e in json.loads(row["evidence_links"])],
            harms=json.loads(row["harms"]),
            status=row["status"],
            created_by=row["created_by"],
            review_notes=[ReviewNote(**n) for n in json.loads(row["review_notes"])],
            tags=json.loads(row["tags"]),
            violation_id=row["violation_id"],
            created_at=row["created_at"],
        )

    def link_memory_entry(self, violation_id: str, memory_entry_id: str) -> bool:
        """Link a memory entry to a violation for later cross reference."""
        self.initialize()
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO violation_memory_links (violation_id, memory_entry_id) VALUES (?, ?)",
                (violation_id, memory_entry_id),
            )
            conn.commit()
            return conn.total_changes > 0
