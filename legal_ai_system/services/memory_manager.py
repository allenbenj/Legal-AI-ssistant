"""
Memory Manager - Service Container Integration Layer

This module provides the MemoryManager class that integrates memory and context
management capabilities with the service container architecture.
"""

import logging
import sqlite3
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import threading

from legal_ai_system.integration_ready.vector_store_enhanced import MemoryStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manager class for memory and context operations
    Handles session persistence, context management, and agent memory
    """
    
    def __init__(
        self,
        db_path: str = "./storage/databases/memory.db",
        max_context_tokens: int = 32000
    ):
        self.db_path = db_path
        self.max_context_tokens = max_context_tokens
        
        self._db_lock = threading.Lock()
        self._memory_store: MemoryStore | None = None
        self._initialized = False
        
        logger.info(f"MemoryManager initialized with db: {db_path}")
    
    async def initialize(self) -> None:
        """Initialize the memory manager"""
        if self._initialized:
            logger.warning("MemoryManager already initialized")
            return
        
        try:
            # Create database directory
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize the memory store
            self._memory_store = MemoryStore(self.db_path)
            
            # Initialize database schema
            await asyncio.get_event_loop().run_in_executor(
                None, self._init_database_schema
            )
            
            self._initialized = True
            logger.info("MemoryManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MemoryManager: {e}")
            raise
    
    def _init_database_schema(self) -> None:
        """Initialize database schema for memory management"""
        with sqlite3.connect(self.db_path) as conn:
            # Create sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    session_name TEXT,
                    context_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Create context_entries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_entries (
                    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    entry_type TEXT,
                    content TEXT,
                    importance_score REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            # Create agent_decisions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_decisions (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT,
                    session_id TEXT,
                    input_summary TEXT,
                    decision TEXT,
                    context_data TEXT,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT
                )
            """)
            
            conn.commit()
    
    async def create_session(
        self,
        session_id: str,
        session_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create a new memory session"""
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized")
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._create_session_sync,
                session_id,
                session_name,
                metadata
            )
            
            logger.info(f"Created session: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {e}")
            raise
    
    def _create_session_sync(
        self,
        session_id: str,
        session_name: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Synchronous session creation"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions 
                (session_id, session_name, context_data, metadata, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                session_id,
                session_name or f"Session {session_id}",
                json.dumps({}),
                json.dumps(metadata or {})
            ))
            conn.commit()
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized")
        
        try:
            session = await asyncio.get_event_loop().run_in_executor(
                None, self._get_session_sync, session_id
            )
            return session
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            raise
    
    def _get_session_sync(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous session retrieval"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM sessions WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'session_id': row['session_id'],
                    'session_name': row['session_name'],
                    'context_data': json.loads(row['context_data'] or '{}'),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'metadata': json.loads(row['metadata'] or '{}')
                }
        return None
    
    async def add_context_entry(
        self,
        session_id: str,
        entry_type: str,
        content: str,
        importance_score: float = 1.0
    ) -> None:
        """Add a context entry to a session"""
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized")
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._add_context_entry_sync,
                session_id,
                entry_type,
                content,
                importance_score
            )
            
            logger.debug(f"Added context entry to session {session_id}: {entry_type}")
            
        except Exception as e:
            logger.error(f"Failed to add context entry: {e}")
            raise
    
    def _add_context_entry_sync(
        self,
        session_id: str,
        entry_type: str,
        content: str,
        importance_score: float
    ) -> None:
        """Synchronous context entry addition"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO context_entries 
                (session_id, entry_type, content, importance_score)
                VALUES (?, ?, ?, ?)
            """, (session_id, entry_type, content, importance_score))
            conn.commit()
    
    async def get_session_context(
        self,
        session_id: str,
        max_entries: int = 50,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Get context entries for a session"""
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized")
        
        try:
            context = await asyncio.get_event_loop().run_in_executor(
                None,
                self._get_session_context_sync,
                session_id,
                max_entries,
                min_importance
            )
            return context
            
        except Exception as e:
            logger.error(f"Failed to get session context: {e}")
            raise
    
    def _get_session_context_sync(
        self,
        session_id: str,
        max_entries: int,
        min_importance: float
    ) -> List[Dict[str, Any]]:
        """Synchronous session context retrieval"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM context_entries 
                WHERE session_id = ? AND importance_score >= ?
                ORDER BY importance_score DESC, created_at DESC
                LIMIT ?
            """, (session_id, min_importance, max_entries))
            
            entries = []
            for row in cursor.fetchall():
                entries.append({
                    'entry_id': row['entry_id'],
                    'entry_type': row['entry_type'],
                    'content': row['content'],
                    'importance_score': row['importance_score'],
                    'created_at': row['created_at']
                })
            
            return entries
    
    async def log_agent_decision(
        self,
        agent_name: str,
        session_id: str,
        input_summary: str,
        decision: str,
        context: Optional[Dict[str, Any]] = None,
        confidence_score: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """Log an agent decision for future reference"""
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized")
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._log_agent_decision_sync,
                agent_name,
                session_id,
                input_summary,
                decision,
                context,
                confidence_score,
                tags
            )
            
            logger.debug(f"Logged decision for {agent_name} in session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to log agent decision: {e}")
            raise
    
    def _log_agent_decision_sync(
        self,
        agent_name: str,
        session_id: str,
        input_summary: str,
        decision: str,
        context: Optional[Dict[str, Any]],
        confidence_score: Optional[float],
        tags: Optional[List[str]]
    ) -> None:
        """Synchronous agent decision logging"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO agent_decisions 
                (agent_name, session_id, input_summary, decision, context_data, confidence_score, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_name,
                session_id,
                input_summary,
                decision,
                json.dumps(context or {}),
                confidence_score,
                json.dumps(tags or [])
            ))
            conn.commit()
    
    async def get_agent_decisions(
        self,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get agent decisions with optional filtering"""
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized")
        
        try:
            decisions = await asyncio.get_event_loop().run_in_executor(
                None,
                self._get_agent_decisions_sync,
                agent_name,
                session_id,
                limit
            )
            return decisions
            
        except Exception as e:
            logger.error(f"Failed to get agent decisions: {e}")
            raise
    
    def _get_agent_decisions_sync(
        self,
        agent_name: Optional[str],
        session_id: Optional[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Synchronous agent decisions retrieval"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM agent_decisions WHERE 1=1"
            params = []
            
            if agent_name:
                query += " AND agent_name = ?"
                params.append(agent_name)
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            
            decisions = []
            for row in cursor.fetchall():
                decisions.append({
                    'decision_id': row['decision_id'],
                    'agent_name': row['agent_name'],
                    'session_id': row['session_id'],
                    'input_summary': row['input_summary'],
                    'decision': row['decision'],
                    'context_data': json.loads(row['context_data'] or '{}'),
                    'confidence_score': row['confidence_score'],
                    'created_at': row['created_at'],
                    'tags': json.loads(row['tags'] or '[]')
                })
            
            return decisions
    
    async def summarize_long_context(self, content: str, max_length: int = 1000) -> str:
        """Summarize long content to fit within context limits"""
        if len(content) <= max_length:
            return content
        
        # Simple truncation with ellipsis
        # In a real implementation, this would use an LLM for intelligent summarization
        return content[:max_length-3] + "..."
    
    async def optimize_context_window(
        self,
        session_id: str,
        target_tokens: int = 16000
    ) -> None:
        """Optimize context window by removing low-importance entries"""
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialized")
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._optimize_context_window_sync,
                session_id,
                target_tokens
            )
            
            logger.info(f"Optimized context window for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to optimize context window: {e}")
            raise
    
    def _optimize_context_window_sync(self, session_id: str, target_tokens: int) -> None:
        """Synchronous context window optimization"""
        with sqlite3.connect(self.db_path) as conn:
            # Remove entries with low importance scores to reduce context size
            # This is a simplified approach - real implementation would estimate tokens
            conn.execute("""
                DELETE FROM context_entries 
                WHERE session_id = ? AND importance_score < 0.3
                AND entry_id NOT IN (
                    SELECT entry_id FROM context_entries 
                    WHERE session_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT 10
                )
            """, (session_id, session_id))
            conn.commit()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        try:
            stats = await asyncio.get_event_loop().run_in_executor(
                None, self._get_statistics_sync
            )
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    def _get_statistics_sync(self) -> Dict[str, Any]:
        """Synchronous statistics retrieval"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Count sessions
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            stats['total_sessions'] = cursor.fetchone()[0]
            
            # Count context entries
            cursor = conn.execute("SELECT COUNT(*) FROM context_entries")
            stats['total_context_entries'] = cursor.fetchone()[0]
            
            # Count agent decisions
            cursor = conn.execute("SELECT COUNT(*) FROM agent_decisions")
            stats['total_agent_decisions'] = cursor.fetchone()[0]
            
            # Get database size
            stats['db_path'] = self.db_path
            try:
                stats['db_size_bytes'] = Path(self.db_path).stat().st_size
            except FileNotFoundError:
                stats['db_size_bytes'] = 0
            
            return stats
    
    async def shutdown(self) -> None:
        """Shutdown the memory manager"""
        # Close any open connections (handled by context managers)
        self._initialized = False
        logger.info("MemoryManager shut down")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of memory manager"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        try:
            stats = await self.get_statistics()
            
            # Test database connection
            await asyncio.get_event_loop().run_in_executor(
                None, self._test_db_connection
            )
            
            return {
                "status": "healthy",
                "statistics": stats,
                "db_path": self.db_path
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _test_db_connection(self) -> None:
        """Test database connection"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("SELECT 1").fetchone()