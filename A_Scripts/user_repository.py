# legal_ai_system/persistence/repositories/user_repository.py
"""
UserRepository for managing persistent storage of User, Session, and AuditLog data.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import json
import uuid

from ....core.detailed_logging import get_detailed_logger, LogCategory, detailed_log_function
from ....core.security_manager import AuthUser, AccessLevel, AuditLogEntry # Use AuthUser alias
from ..enhanced_persistence import ConnectionPool, TransactionManager, DatabaseError # Use existing persistence components

user_repo_logger = get_detailed_logger("UserRepository", LogCategory.DATABASE)

class UserRepository:
    def __init__(self, connection_pool: ConnectionPool):
        self.pool = connection_pool
        self.transaction_manager = TransactionManager(connection_pool)
        self.logger = user_repo_logger

    @detailed_log_function(LogCategory.DATABASE)
    async def add_user_async(self, user: AuthUser) -> None:
        self.logger.info("Adding new user to database.", parameters={'username': user.username, 'user_id': user.user_id})
        try:
            async with self.transaction_manager.transaction() as conn:
                await conn.execute("""
                    INSERT INTO system_users (user_id, username, email, password_hash, salt, access_level, created_at, is_active)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, user.user_id, user.username, user.email, user.password_hash, user.salt, 
                     user.access_level.value, user.created_at, user.is_active)
            self.logger.info("User added successfully.", parameters={'user_id': user.user_id})
        except Exception as e: # Catch specific asyncpg.UniqueViolationError if possible
            self.logger.error("Failed to add user to database.", exception=e)
            raise DatabaseError(f"Error adding user {user.username}", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def get_user_by_username_async(self, username: str) -> Optional[AuthUser]:
        self.logger.debug("Fetching user by username.", parameters={'username': username})
        try:
            async with self.pool.get_pg_connection() as conn:
                row = await conn.fetchrow("SELECT * FROM system_users WHERE username = $1", username)
                return self._row_to_auth_user(row) if row else None
        except Exception as e:
            self.logger.error("Failed to fetch user by username.", exception=e)
            raise DatabaseError(f"Error fetching user {username}", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def get_user_by_id_async(self, user_id: str) -> Optional[AuthUser]:
        self.logger.debug("Fetching user by ID.", parameters={'user_id': user_id})
        try:
            async with self.pool.get_pg_connection() as conn:
                row = await conn.fetchrow("SELECT * FROM system_users WHERE user_id = $1", user_id)
                return self._row_to_auth_user(row) if row else None
        except Exception as e:
            self.logger.error("Failed to fetch user by ID.", exception=e)
            raise DatabaseError(f"Error fetching user ID {user_id}", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def update_user_auth_status_async(self, user_id: str, failed_attempts: int, 
                                           locked_until: Optional[datetime], 
                                           last_login: Optional[datetime] = None) -> None:
        self.logger.info("Updating user auth status.", parameters={'user_id': user_id, 'failed_attempts': failed_attempts})
        try:
            async with self.transaction_manager.transaction() as conn:
                await conn.execute("""
                    UPDATE system_users 
                    SET failed_attempts = $1, locked_until = $2, last_login = $3, updated_at = $4
                    WHERE user_id = $5
                """, failed_attempts, locked_until, last_login, datetime.now(timezone.utc), user_id)
            self.logger.info("User auth status updated.", parameters={'user_id': user_id})
        except Exception as e:
            self.logger.error("Failed to update user auth status.", exception=e)
            raise DatabaseError(f"Error updating auth status for user {user_id}", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def create_session_async(self, session_token: str, session_data: Dict[str, Any]) -> None:
        self.logger.info("Creating persistent session.", parameters={'token_preview': session_token[:8]+"...", 'user_id': session_data.get('user_id')})
        try:
            async with self.transaction_manager.transaction() as conn:
                await conn.execute("""
                    INSERT INTO user_sessions_persistent (session_token, user_id, created_at, expires_at, ip_address, user_agent, is_valid)
                    VALUES ($1, $2, $3, $4, $5, $6, TRUE)
                """, session_token, session_data['user_id'], session_data['created_at'], 
                     session_data['expires_at'], session_data.get('ip_address'), session_data.get('user_agent'))
            self.logger.info("Persistent session created.", parameters={'token_preview': session_token[:8]+"..."})
        except Exception as e:
            self.logger.error("Failed to create persistent session.", exception=e)
            raise DatabaseError(f"Error creating session for user {session_data.get('user_id')}", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def get_session_async(self, session_token: str) -> Optional[Dict[str, Any]]:
        self.logger.debug("Fetching persistent session.", parameters={'token_preview': session_token[:8]+"..."})
        try:
            async with self.pool.get_pg_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT user_id, created_at, expires_at, ip_address, user_agent 
                    FROM user_sessions_persistent 
                    WHERE session_token = $1 AND is_valid = TRUE AND expires_at > NOW()
                """, session_token)
                return dict(row) if row else None
        except Exception as e:
            self.logger.error("Failed to fetch persistent session.", exception=e)
            raise DatabaseError(f"Error fetching session token {session_token[:8]}...", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def invalidate_session_async(self, session_token: str) -> None:
        self.logger.info("Invalidating persistent session.", parameters={'token_preview': session_token[:8]+"..."})
        try:
            async with self.transaction_manager.transaction() as conn:
                await conn.execute("UPDATE user_sessions_persistent SET is_valid = FALSE WHERE session_token = $1", session_token)
            self.logger.info("Persistent session invalidated.", parameters={'token_preview': session_token[:8]+"..."})
        except Exception as e:
            self.logger.error("Failed to invalidate persistent session.", exception=e)
            raise DatabaseError(f"Error invalidating session token {session_token[:8]}...", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def add_audit_logs_batch_async(self, logs: List[AuditLogEntry]) -> None:
        if not logs: return
        self.logger.info(f"Adding batch of {len(logs)} audit logs to database.")
        try:
            log_data_tuples = [
                (l.entry_id, l.timestamp, l.user_id, l.action, l.resource, 
                 json.dumps(l.details, default=str), l.ip_address, l.user_agent, l.status)
                for l in logs
            ]
            async with self.transaction_manager.transaction() as conn: # Batch insert in one transaction
                await conn.executemany("""
                    INSERT INTO system_audit_log (log_id, timestamp, user_id, action, resource, details, ip_address, user_agent, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, log_data_tuples)
            self.logger.info("Batch audit logs added successfully.", parameters={'count': len(logs)})
        except Exception as e:
            self.logger.error("Failed to add batch audit logs.", exception=e)
            # Consider a fallback to file logging for critical audit logs if DB fails
            raise DatabaseError("Error adding batch audit logs.", cause=e)

    def _row_to_auth_user(self, row: Optional[asyncpg.Record]) -> Optional[AuthUser]: # Changed type hint
        if not row: return None
        return AuthUser(
            user_id=row['user_id'], username=row['username'], email=row['email'],
            password_hash=row['password_hash'], salt=row['salt'],
            access_level=AccessLevel(row['access_level']), # Convert string back to Enum
            created_at=row['created_at'], last_login=row.get('last_login'),
            is_active=row['is_active'], failed_attempts=row['failed_attempts'],
            locked_until=row.get('locked_until')
        )