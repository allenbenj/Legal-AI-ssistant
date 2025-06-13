from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from legal_ai_system.services.security_manager import (
    AccessLevel,
    AuditLogEntry,
    User as AuthUser,
)


class SQLiteUserRepository:
    """SQLite-backed replacement for the original PostgreSQL UserRepository."""

    def __init__(self, db_path: str | Path = "./storage/databases/user_auth.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create required tables if they don't exist."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS system_users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    access_level TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active INTEGER DEFAULT 1,
                    failed_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS user_sessions_persistent (
                    session_token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    is_valid INTEGER DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES system_users(user_id)
                );
                CREATE TABLE IF NOT EXISTS system_audit_log (
                    log_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    user_id TEXT,
                    action TEXT,
                    resource TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    status TEXT
                );
                """
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Public API mirroring the original UserRepository
    # ------------------------------------------------------------------
    async def add_user_async(self, user: AuthUser) -> None:
        await asyncio.get_event_loop().run_in_executor(
            None, self._add_user_sync, user
        )

    def _add_user_sync(self, user: AuthUser) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO system_users (
                    user_id, username, email, password_hash, salt,
                    access_level, created_at, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user.user_id,
                    user.username,
                    user.email,
                    user.password_hash,
                    user.salt,
                    user.access_level.value,
                    user.created_at,
                    int(user.is_active),
                ),
            )
            conn.commit()

    async def get_user_by_username_async(self, username: str) -> Optional[AuthUser]:
        row = await asyncio.get_event_loop().run_in_executor(
            None, self._get_user_by_username_sync, username
        )
        return self._row_to_auth_user(row) if row else None

    def _get_user_by_username_sync(self, username: str) -> Optional[sqlite3.Row]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM system_users WHERE username = ?", (username,)
            )
            return cursor.fetchone()

    async def get_user_by_id_async(self, user_id: str) -> Optional[AuthUser]:
        row = await asyncio.get_event_loop().run_in_executor(
            None, self._get_user_by_id_sync, user_id
        )
        return self._row_to_auth_user(row) if row else None

    def _get_user_by_id_sync(self, user_id: str) -> Optional[sqlite3.Row]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM system_users WHERE user_id = ?", (user_id,)
            )
            return cursor.fetchone()

    async def update_user_auth_status_async(
        self,
        user_id: str,
        failed_attempts: int,
        locked_until: Optional[datetime],
        last_login: Optional[datetime] = None,
    ) -> None:
        await asyncio.get_event_loop().run_in_executor(
            None,
            self._update_user_auth_status_sync,
            user_id,
            failed_attempts,
            locked_until,
            last_login,
        )

    def _update_user_auth_status_sync(
        self,
        user_id: str,
        failed_attempts: int,
        locked_until: Optional[datetime],
        last_login: Optional[datetime],
    ) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE system_users
                SET failed_attempts = ?, locked_until = ?, last_login = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (
                    failed_attempts,
                    locked_until,
                    last_login,
                    datetime.utcnow(),
                    user_id,
                ),
            )
            conn.commit()

    async def create_session_async(self, session_token: str, session_data: Dict[str, Any]) -> None:
        await asyncio.get_event_loop().run_in_executor(
            None, self._create_session_sync, session_token, session_data
        )

    def _create_session_sync(self, token: str, data: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_sessions_persistent (
                    session_token, user_id, created_at, expires_at, ip_address, user_agent, is_valid
                ) VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    token,
                    data["user_id"],
                    data["created_at"],
                    data["expires_at"],
                    data.get("ip_address"),
                    data.get("user_agent"),
                ),
            )
            conn.commit()

    async def get_session_async(self, session_token: str) -> Optional[Dict[str, Any]]:
        row = await asyncio.get_event_loop().run_in_executor(
            None, self._get_session_sync, session_token
        )
        if not row:
            return None
        return {
            "user_id": row["user_id"],
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
            "ip_address": row["ip_address"],
            "user_agent": row["user_agent"],
        }

    def _get_session_sync(self, token: str) -> Optional[sqlite3.Row]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT user_id, created_at, expires_at, ip_address, user_agent
                FROM user_sessions_persistent
                WHERE session_token = ? AND is_valid = 1 AND expires_at > ?
                """,
                (token, datetime.utcnow()),
            )
            return cursor.fetchone()

    async def invalidate_session_async(self, session_token: str) -> None:
        await asyncio.get_event_loop().run_in_executor(
            None, self._invalidate_session_sync, session_token
        )

    def _invalidate_session_sync(self, token: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE user_sessions_persistent SET is_valid = 0 WHERE session_token = ?",
                (token,),
            )
            conn.commit()

    async def add_audit_logs_batch_async(self, logs: List[AuditLogEntry]) -> None:
        if not logs:
            return
        await asyncio.get_event_loop().run_in_executor(
            None, self._add_audit_logs_sync, logs
        )

    def _add_audit_logs_sync(self, logs: List[AuditLogEntry]) -> None:
        with self._get_connection() as conn:
            data = [
                (
                    l.entry_id,
                    l.timestamp,
                    l.user_id,
                    l.action,
                    l.resource,
                    json.dumps(l.details, default=str),
                    l.ip_address,
                    l.user_agent,
                    l.status,
                )
                for l in logs
            ]
            conn.executemany(
                """
                INSERT INTO system_audit_log (
                    log_id, timestamp, user_id, action, resource,
                    details, ip_address, user_agent, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                data,
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _row_to_auth_user(self, row: sqlite3.Row) -> AuthUser:
        return AuthUser(
            user_id=row["user_id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            salt=row["salt"],
            access_level=AccessLevel(row["access_level"]),
            created_at=row["created_at"],
            last_login=row["last_login"],
            is_active=bool(row["is_active"]),
            failed_attempts=row["failed_attempts"],
            locked_until=row["locked_until"],
        )

__all__ = ["SQLiteUserRepository"]

