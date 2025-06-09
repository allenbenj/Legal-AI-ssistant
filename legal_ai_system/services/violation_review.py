"""Violation Review management module."""

import sqlite3
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
import uuid

logger = logging.getLogger(__name__)


@dataclass
class BarViolation:
    """Professional conduct violation reference."""
    rule: str
    description: str
    verified_by: str
    verified_at: str


@dataclass
class ActorInfo:
    """Information about the actor committing the violation."""
    name: str
    role: str
    bar_violations: List[BarViolation] = field(default_factory=list)


@dataclass
class MotionRecommendation:
    """Recommended motion in response to the violation."""
    type: str
    rule: str
    commentary: str


@dataclass
class EthicsCheck:
    """Record of ethics validation for this violation."""
    verified_by: str
    comment: str
    timestamp: str


@dataclass
class ViolationReviewEntry:
    """Dataclass representing a single violation review record."""
    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str = ""
    violation_type: str = ""
    statute: str = ""
    jurisdiction: str = ""
    actor: ActorInfo = field(default_factory=ActorInfo)
    related_statements: List[str] = field(default_factory=list)
    evidence_links: List[Dict[str, Any]] = field(default_factory=list)
    harms: List[str] = field(default_factory=list)
    motion_recommendation: Optional[MotionRecommendation] = None
    status: str = "Open"
    created_by: str = ""
    ethics_check: Optional[EthicsCheck] = None
    tags: List[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ViolationReviewEntry":
        actor = ActorInfo(
            name=row["actor_name"],
            role=row["actor_role"],
            bar_violations=[
                BarViolation(**bv) for bv in json.loads(row["bar_violations"] or "[]")
            ],
        )
        motion = None
        if row["motion_type"]:
            motion = MotionRecommendation(
                type=row["motion_type"],
                rule=row["motion_rule"],
                commentary=row["motion_commentary"],
            )
        ethics = None
        if row["ethics_verified_by"]:
            ethics = EthicsCheck(
                verified_by=row["ethics_verified_by"],
                comment=row["ethics_comment"],
                timestamp=row["ethics_timestamp"],
            )
        return cls(
            violation_id=row["violation_id"],
            case_id=row["case_id"],
            violation_type=row["violation_type"],
            statute=row["statute"],
            jurisdiction=row["jurisdiction"],
            actor=actor,
            related_statements=json.loads(row["related_statements"] or "[]"),
            evidence_links=json.loads(row["evidence_links"] or "[]"),
            harms=json.loads(row["harms"] or "[]"),
            motion_recommendation=motion,
            status=row["status"],
            created_by=row["created_by"],
            ethics_check=ethics,
            tags=json.loads(row["tags"] or "[]"),
            created_at=row["created_at"],
        )

    def to_db_tuple(self) -> tuple:
        return (
            self.violation_id,
            self.case_id,
            self.violation_type,
            self.statute,
            self.jurisdiction,
            self.actor.name,
            self.actor.role,
            json.dumps([asdict(bv) for bv in self.actor.bar_violations]),
            json.dumps(self.related_statements),
            json.dumps(self.evidence_links),
            json.dumps(self.harms),
            (
                self.motion_recommendation.type
                if self.motion_recommendation
                else None
            ),
            (
                self.motion_recommendation.rule
                if self.motion_recommendation
                else None
            ),
            (
                self.motion_recommendation.commentary
                if self.motion_recommendation
                else None
            ),
            self.status,
            self.created_by,
            self.ethics_check.verified_by if self.ethics_check else None,
            self.ethics_check.comment if self.ethics_check else None,
            self.ethics_check.timestamp if self.ethics_check else None,
            json.dumps(self.tags),
            self.created_at,
        )


class ViolationReviewManager:
    """SQLite-backed manager for violation review records."""

    def __init__(self, db_path: str = "./storage/databases/violation_review.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _initialize_database(self) -> None:
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
                    bar_violations TEXT,
                    related_statements TEXT,
                    evidence_links TEXT,
                    harms TEXT,
                    motion_type TEXT,
                    motion_rule TEXT,
                    motion_commentary TEXT,
                    status TEXT,
                    created_by TEXT,
                    ethics_verified_by TEXT,
                    ethics_comment TEXT,
                    ethics_timestamp TEXT,
                    tags TEXT,
                    created_at TEXT
                )
                """
            )
            conn.commit()

    def insert_violation(self, entry: ViolationReviewEntry) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO violations (
                        violation_id, case_id, violation_type, statute, jurisdiction,
                        actor_name, actor_role, bar_violations, related_statements,
                        evidence_links, harms, motion_type, motion_rule,
                        motion_commentary, status, created_by,
                        ethics_verified_by, ethics_comment, ethics_timestamp, tags,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    entry.to_db_tuple(),
                )
                conn.commit()
            return True
        except Exception as exc:  # pragma: no cover - simple logging
            logger.error(f"Failed to insert violation: {exc}")
            return False

    def get_violation(self, violation_id: str) -> Optional[ViolationReviewEntry]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM violations WHERE violation_id = ?",
                (violation_id,),
            )
            row = cursor.fetchone()
            return ViolationReviewEntry.from_row(row) if row else None

    def search_violations(self, filters: Optional[Dict[str, Any]] = None) -> List[ViolationReviewEntry]:
        query = "SELECT * FROM violations WHERE 1=1"
        params: List[Any] = []
        if filters:
            if filters.get("case_id"):
                query += " AND case_id = ?"
                params.append(filters["case_id"])
            if filters.get("violation_type"):
                query += " AND violation_type = ?"
                params.append(filters["violation_type"])
            if filters.get("status"):
                query += " AND status = ?"
                params.append(filters["status"])
        query += " ORDER BY created_at DESC"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [ViolationReviewEntry.from_row(r) for r in rows]

    def update_violation(self, violation_id: str, updates: Dict[str, Any]) -> bool:
        if not updates:
            return False
        allowed = {
            "status",
            "motion_type",
            "motion_rule",
            "motion_commentary",
            "ethics_verified_by",
            "ethics_comment",
            "ethics_timestamp",
        }
        set_clauses = []
        params: List[Any] = []
        for key, value in updates.items():
            if key not in allowed:
                continue
            set_clauses.append(f"{key} = ?")
            params.append(value)
        if not set_clauses:
            return False
        params.append(violation_id)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    f"UPDATE violations SET {', '.join(set_clauses)} WHERE violation_id = ?",
                    params,
                )
                conn.commit()
                return conn.total_changes > 0
        except Exception as exc:  # pragma: no cover - simple logging
            logger.error(f"Failed to update violation {violation_id}: {exc}")
            return False

