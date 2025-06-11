# legal_ai_system/memory/unified_memory_manager.py

"""
Unified Memory Manager - Consolidated Memory Components
====================================================
Single source of truth for all memory operations in the Legal AI System.
Combines agent memory, session persistence, knowledge graph elements for sessions,
and context window management.
"""

import asyncio
import json
import sqlite3
import threading
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import hashlib

# Use detailed_logging
from ..core.detailed_logging import (
    get_detailed_logger,
    LogCategory,
    detailed_log_function,
)

# Import exceptions
from ..core.unified_exceptions import MemoryManagerError

# Import constants if needed, e.g., for max_context_tokens default
from ..core.constants import Constants
from .config_models import UnifiedMemoryManagerConfig


# Initialize loggers for this module and its sub-components
umm_logger = get_detailed_logger("UnifiedMemoryManager", LogCategory.DATABASE)
agent_mem_logger = get_detailed_logger(
    "UnifiedMemoryManager.AgentMemory", LogCategory.DATABASE
)
session_mem_logger = get_detailed_logger(
    "UnifiedMemoryManager.SessionMemory", LogCategory.DATABASE
)
context_mem_logger = get_detailed_logger(
    "UnifiedMemoryManager.ContextWindowMemory", LogCategory.DATABASE
)

# Reserved name used for shared session knowledge entries
SHARED_AGENT_NAME = "__shared__"


class MemoryType(Enum):
    """Types of memory storage managed by UMM."""

    AGENT_SPECIFIC = (
        "agent_specific"  # Memory unique to an agent's internal state for a task/doc
    )
    SESSION_KNOWLEDGE = (
        "session_knowledge"  # Entities, facts, observations relevant to a user session
    )
    CONTEXT_WINDOW = "context_window"  # Short-term conversational/operational context
    # DOCUMENT_CACHE = "document_cache"     # Potentially for cached processed document parts
    # ENTITY_CACHE = "entity_cache"         # Potentially for cached resolved entities


@dataclass
class MemoryEntry:
    """Standardized memory entry (conceptual)."""

    id: str  # Unique ID for the entry
    memory_type: MemoryType
    key: str  # Primary key/identifier for the data within its type (e.g., entity_name, agent_key)
    value: Any  # The actual data stored (can be JSON string)
    session_id: Optional[str] = None  # Link to a user/operational session
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    importance_score: float = 0.5  # For ranking/pruning


class UnifiedMemoryManager:
    """Manages various types of memory for the Legal AI System."""

    @detailed_log_function(LogCategory.DATABASE)
    def __init__(
        self,
        db_path_str: str = "./storage/databases/unified_memory.db",
        max_context_tokens_config: int = Constants.Size.MAX_CONTEXT_TOKENS,
        service_config: Optional[UnifiedMemoryManagerConfig] = None,
    ):
        umm_logger.info("=== INITIALIZING UNIFIED MEMORY MANAGER ===")

        self.config = service_config or UnifiedMemoryManagerConfig()
        self.db_path = Path(db_path_str)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.max_context_tokens = max_context_tokens_config

        self._lock = threading.RLock()  # Thread safety for DB operations
        self._initialized = False

        # Performance tracking (basic)
        self._operation_counts: Dict[str, int] = defaultdict(int)
        self._last_operation_time: Optional[datetime] = None

    @detailed_log_function(LogCategory.DATABASE)
    async def initialize(self):
        """Initialize the UnifiedMemoryManager and its database schema."""
        if self._initialized:
            umm_logger.warning("UnifiedMemoryManager already initialized.")
            return self

        umm_logger.info("Starting UnifiedMemoryManager initialization.")
        try:
            # Run synchronous DB init in an executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._initialize_database_schema_sync)

            self._initialized = True
            umm_logger.info(
                "UnifiedMemoryManager initialized successfully.",
                parameters={"db_path": str(self.db_path)},
            )
        except Exception as e:
            umm_logger.critical(
                "UnifiedMemoryManager initialization failed.", exception=e
            )
            self._initialized = False
            raise MemoryManagerError(
                "Failed to initialize UnifiedMemoryManager database.", cause=e
            )
        return self

    def _initialize_database_schema_sync(self):
        """Synchronously initializes the SQLite database schema."""
        umm_logger.debug(
            "Initializing SQLite database schema.",
            parameters={"db_path": str(self.db_path)},
        )
        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
                conn.executescript(
                    """
                    -- Agent-specific memory
                    CREATE TABLE IF NOT EXISTS agent_memory (
                        id TEXT PRIMARY KEY, -- Composite: session_id_agent_name_memory_type_key_hash
                        session_id TEXT, -- Can be doc_id or actual session
                        agent_name TEXT NOT NULL,
                        memory_type TEXT NOT NULL DEFAULT 'agent_specific',
                        memory_key TEXT NOT NULL,
                        memory_value TEXT, -- JSON serialized
                        metadata TEXT, -- JSON serialized
                        memory_type TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        importance_score REAL DEFAULT 0.5
                    );
                    CREATE INDEX IF NOT EXISTS idx_agent_memory_session_agent ON agent_memory(session_id, agent_name);
                    CREATE INDEX IF NOT EXISTS idx_agent_memory_key ON agent_memory(memory_key);
                    CREATE INDEX IF NOT EXISTS idx_agent_memory_type ON agent_memory(memory_type);

                    -- Session-scoped Knowledge (inspired by ClaudeMemoryStore)
                    CREATE TABLE IF NOT EXISTS session_entities (
                        entity_id TEXT PRIMARY KEY, -- Unique ID for the session entity instance
                        session_id TEXT NOT NULL,
                        canonical_name TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        attributes TEXT, -- JSON
                        confidence_score REAL,
                        memory_type TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        source_description TEXT -- e.g., "LLM extraction", "User input"
                    );
                    CREATE INDEX IF NOT EXISTS idx_session_entities_session_type_name ON session_entities(session_id, entity_type, canonical_name);

                    CREATE TABLE IF NOT EXISTS session_observations (
                        observation_id TEXT PRIMARY KEY,
                        session_entity_id TEXT NOT NULL, -- Link to session_entities.entity_id
                        session_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        importance_score REAL DEFAULT 0.5,
                        memory_type TEXT NOT NULL,
                        source TEXT,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_entity_id) REFERENCES session_entities (entity_id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_session_obs_entity ON session_observations(session_entity_id);

                    CREATE TABLE IF NOT EXISTS session_relationships (
                        relationship_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        source_session_entity_id TEXT NOT NULL,
                        target_session_entity_id TEXT NOT NULL,
                        relationship_type TEXT NOT NULL,
                        properties TEXT, -- JSON
                        confidence_score REAL,
                        memory_type TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (source_session_entity_id) REFERENCES session_entities (entity_id) ON DELETE CASCADE,
                        FOREIGN KEY (target_session_entity_id) REFERENCES session_entities (entity_id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_session_rels_session_source_target ON session_relationships(session_id, source_session_entity_id, target_session_entity_id);
                    
                    -- Context Window Entries (from original MemoryManager)
                    CREATE TABLE IF NOT EXISTS context_window_entries (
                        entry_id TEXT PRIMARY KEY, -- UUID
                        session_id TEXT NOT NULL,
                        entry_type TEXT NOT NULL, -- e.g., 'user_query', 'llm_response', 'document_chunk'
                        content TEXT NOT NULL,    -- Can be JSON
                        token_count INTEGER,  -- Estimated token count
                        importance_score REAL DEFAULT 0.5,
                        memory_type TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT -- JSON
                    );
                    CREATE INDEX IF NOT EXISTS idx_context_session_created ON context_window_entries(session_id, created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_context_importance ON context_window_entries(session_id, importance_score DESC);

                    -- Agent Decisions Log (from original MemoryManager)
                    CREATE TABLE IF NOT EXISTS agent_decisions_log (
                        decision_id TEXT PRIMARY KEY, -- UUID
                        agent_name TEXT NOT NULL,
                        session_id TEXT,
                        input_summary TEXT,
                        decision_details TEXT, -- JSON serialized decision
                        context_used TEXT, -- JSON summary of context
                        confidence_score REAL,
                        tags TEXT, -- JSON list of tags
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_agent_decisions_agent_session ON agent_decisions_log(agent_name, session_id, created_at DESC);

                    -- Tag Learning Stats
                    CREATE TABLE IF NOT EXISTS tag_learning_stats (
                        tag_text TEXT PRIMARY KEY,
                        correct_count INTEGER DEFAULT 0,
                        incorrect_count INTEGER DEFAULT 0,
                        suggested_count INTEGER DEFAULT 0,
                        last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_tag_stats_last_updated ON tag_learning_stats(last_updated DESC);

                    -- Pattern Learning Stats
                    CREATE TABLE IF NOT EXISTS pattern_learning_stats (
                        pattern_hash TEXT PRIMARY KEY, -- MD5 hash of the pattern_regex
                        pattern_regex TEXT UNIQUE NOT NULL,
                        effectiveness_score REAL DEFAULT 0.5, -- e.g., precision or F1
                        usage_count INTEGER DEFAULT 0,
                        last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_pattern_stats_effectiveness ON pattern_learning_stats(effectiveness_score DESC);
                """
                )
                conn.commit()
            umm_logger.info("Unified memory database schema initialized/verified.")
        except sqlite3.Error as e:
            umm_logger.critical(
                "SQLite error during schema initialization.", exception=e
            )
            raise MemoryManagerError("Database schema initialization failed.", cause=e)

    def _get_db_connection(self) -> sqlite3.Connection:
        """Gets a new SQLite connection."""
        try:
            return sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        except sqlite3.Error as e:
            umm_logger.error("Failed to connect to SQLite database.", exception=e)
            raise MemoryManagerError("Database connection failed.", cause=e)

    def _record_op(self, op_name: str):
        """Records an operation for basic performance tracking."""
        with self._lock:
            self._operation_counts[op_name] = self._operation_counts.get(op_name, 0) + 1
            self._last_operation_time = datetime.now(timezone.utc)

    # --- Agent Specific Memory ---
    @detailed_log_function(LogCategory.DATABASE)
    async def store_agent_memory(
        self,
        session_id: str,
        agent_name: str,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        memory_type: MemoryType = MemoryType.AGENT_SPECIFIC,
    ) -> MemoryEntry:
        agent_mem_logger.info(
            "Storing agent memory.",
            parameters={"session": session_id, "agent": agent_name, "key": key},
        )
        self._record_op("store_agent_memory")

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            memory_type=memory_type,
            key=key,
            value=value,
            session_id=session_id,
            metadata=metadata or {},
            importance_score=importance,
        )
        value_json = json.dumps(entry.value)
        metadata_json = json.dumps(entry.metadata)
        now_iso = datetime.now(timezone.utc).isoformat()

        def _store_sync():
            with self._lock, self._get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO agent_memory

                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entry.id,
                        entry.session_id,
                        agent_name,
                        value_json,
                        metadata_json,
                        entry.memory_type.value,
                        now_iso,
                        now_iso,
                        entry.importance_score,
                    ),
                )
                conn.commit()

        try:
            await asyncio.get_event_loop().run_in_executor(None, _store_sync)
            agent_mem_logger.debug(
                "Agent memory stored successfully.", parameters={"id": entry.id}
            )
            return entry
        except Exception as e:
            agent_mem_logger.error("Failed to store agent memory.", exception=e)
            raise MemoryManagerError("Failed to store agent memory.", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def retrieve_agent_memory(
        self,
        session_id: str,
        agent_name: str,
        key: str,
        memory_type: MemoryType = MemoryType.AGENT_SPECIFIC,
    ) -> Optional[MemoryEntry]:
        agent_mem_logger.debug(
            "Retrieving agent memory.",
            parameters={"session": session_id, "agent": agent_name, "key": key},
        )
        self._record_op("retrieve_agent_memory")

        def _retrieve_sync():
            with self._lock, self._get_db_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                )
                row = cursor.fetchone()
                if row:
                    return MemoryEntry(
                        id=row["id"],
                        memory_type=MemoryType(row["memory_type"]),
                        key=row["memory_key"],
                        value=json.loads(row["memory_value"]),
                        session_id=row["session_id"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                        importance_score=row["importance_score"],
                    )
                return None

        try:
            value = await asyncio.get_event_loop().run_in_executor(None, _retrieve_sync)
            if value is not None:
                agent_mem_logger.debug(
                    "Agent memory retrieved.", parameters={"key": key}
                )
            else:
                agent_mem_logger.debug(
                    "Agent memory not found.", parameters={"key": key}
                )
            return value
        except Exception as e:
            agent_mem_logger.error("Failed to retrieve agent memory.", exception=e)
            raise MemoryManagerError("Failed to retrieve agent memory.", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def store_shared_memory(
        self,
        session_id: str,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> str:
        """Store session knowledge shared across agents."""
        return await self.store_agent_memory(
            session_id=session_id,
            agent_name=SHARED_AGENT_NAME,
            key=key,
            value=value,
            metadata=metadata,
            importance=importance,
            memory_type=MemoryType.SESSION_KNOWLEDGE,
        )

    @detailed_log_function(LogCategory.DATABASE)
    async def retrieve_shared_memory(
        self,
        session_id: str,
        key: str,
    ) -> Optional[Any]:
        """Retrieve shared session knowledge without agent restriction."""

        def _retrieve_sync():
            with self._lock, self._get_db_connection() as conn:
                cursor = conn.execute(
                    "SELECT memory_value FROM agent_memory WHERE session_id=? AND memory_type=? AND memory_key=?",
                    (session_id, MemoryType.SESSION_KNOWLEDGE.value, key),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None

        try:
            value = await asyncio.get_event_loop().run_in_executor(None, _retrieve_sync)
            return value
        except Exception as e:
            agent_mem_logger.error("Failed to retrieve shared memory.", exception=e)
            raise MemoryManagerError("Failed to retrieve shared memory.", cause=e)

    # --- Session Knowledge (Claude-like Memory) ---
    @detailed_log_function(LogCategory.DATABASE)
    async def store_session_entity(
        self,
        session_id: str,
        name: str,
        entity_type: str,
        attributes: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        source: str = "system",
        memory_type: MemoryType = MemoryType.SESSION_KNOWLEDGE,
    ) -> str:
        session_mem_logger.info(
            "Storing session entity.",
            parameters={"session": session_id, "name": name, "type": entity_type},
        )
        self._record_op("store_session_entity")

        entity_id = hashlib.md5(
            f"{session_id}:{entity_type}:{name}".encode()
        ).hexdigest()
        attr_json = json.dumps(attributes, default=str) if attributes else "{}"
        now_iso = datetime.now(timezone.utc).isoformat()

        def _store_sync():
            with self._lock, self._get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO session_entities
                        (entity_id, session_id, canonical_name, entity_type, attributes, confidence_score, memory_type, created_at, updated_at, source_description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entity_id,
                        session_id,
                        name,
                        entity_type,
                        attr_json,
                        confidence,
                        memory_type.value,
                        now_iso,
                        now_iso,
                        source,
                    ),
                )
                conn.commit()

        try:
            await asyncio.get_event_loop().run_in_executor(None, _store_sync)
            session_mem_logger.debug(
                "Session entity stored.", parameters={"id": entity_id}
            )
            return entity_id
        except Exception as e:
            session_mem_logger.error("Failed to store session entity.", exception=e)
            raise MemoryManagerError("Failed to store session entity.", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def add_session_observation(
        self,
        session_id: str,
        entity_id: str,
        content: str,
        importance: float = 0.5,
        source: str = "system",
        memory_type: MemoryType = MemoryType.SESSION_KNOWLEDGE,
    ) -> str:
        session_mem_logger.info(
            "Adding session observation.",
            parameters={"session": session_id, "entity": entity_id},
        )
        self._record_op("add_session_observation")

        observation_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        def _add_sync():
            with self._lock, self._get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO session_observations
                        (observation_id, session_entity_id, session_id, content, importance_score, memory_type, source, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        observation_id,
                        entity_id,
                        session_id,
                        content,
                        importance,
                        memory_type.value,
                        source,
                        now_iso,
                    ),
                )
                conn.commit()

        try:
            await asyncio.get_event_loop().run_in_executor(None, _add_sync)
            session_mem_logger.debug(
                "Session observation added.", parameters={"id": observation_id}
            )
            return observation_id
        except Exception as e:
            session_mem_logger.error("Failed to add session observation.", exception=e)
            raise MemoryManagerError("Failed to add session observation.", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def create_session_relationship(
        self,
        session_id: str,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        memory_type: MemoryType = MemoryType.SESSION_KNOWLEDGE,
    ) -> str:
        session_mem_logger.info(
            "Creating session relationship.",
            parameters={"session": session_id, "type": relationship_type},
        )
        self._record_op("create_session_relationship")

        relationship_id = str(uuid.uuid4())
        props_json = json.dumps(properties, default=str) if properties else "{}"
        now_iso = datetime.now(timezone.utc).isoformat()

        def _create_sync():
            with self._lock, self._get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO session_relationships
                        (relationship_id, session_id, source_session_entity_id, target_session_entity_id,
                         relationship_type, properties, confidence_score, memory_type, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        relationship_id,
                        session_id,
                        source_entity_id,
                        target_entity_id,
                        relationship_type,
                        props_json,
                        confidence,
                        memory_type.value,
                        now_iso,
                    ),
                )
                conn.commit()

        try:
            await asyncio.get_event_loop().run_in_executor(None, _create_sync)
            session_mem_logger.debug(
                "Session relationship created.", parameters={"id": relationship_id}
            )
            return relationship_id
        except Exception as e:
            session_mem_logger.error(
                "Failed to create session relationship.", exception=e
            )
            raise MemoryManagerError("Failed to create session relationship.", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def get_session_entity(
        self, session_id: str, entity_type: str, name: str
    ) -> Optional[Dict[str, Any]]:
        session_mem_logger.debug(
            "Getting session entity.",
            parameters={"session": session_id, "type": entity_type, "name": name},
        )
        self._record_op("get_session_entity")

        def _get_sync():
            with self._lock, self._get_db_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM session_entities 
                    WHERE session_id=? AND entity_type=? AND canonical_name=?
                """,
                    (session_id, entity_type, name),
                )
                row = cursor.fetchone()
                if row:
                    entity_dict = dict(row)
                    entity_dict["attributes"] = (
                        json.loads(entity_dict["attributes"])
                        if entity_dict["attributes"]
                        else {}
                    )
                    return entity_dict
                return None

        try:
            entity = await asyncio.get_event_loop().run_in_executor(None, _get_sync)
            if entity:
                session_mem_logger.debug(
                    "Session entity found.", parameters={"name": name}
                )
            else:
                session_mem_logger.debug(
                    "Session entity not found.", parameters={"name": name}
                )
            return entity
        except Exception as e:
            session_mem_logger.error("Failed to get session entity.", exception=e)
            raise MemoryManagerError("Failed to get session entity.", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def get_session_entities(
        self, session_id: str, entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        session_mem_logger.debug(
            "Getting session entities.",
            parameters={"session": session_id, "type": entity_type},
        )
        self._record_op("get_session_entities")

        def _get_sync():
            with self._lock, self._get_db_connection() as conn:
                conn.row_factory = sqlite3.Row
                if entity_type:
                    cursor = conn.execute(
                        """
                        SELECT * FROM session_entities 
                        WHERE session_id=? AND entity_type=?
                        ORDER BY confidence_score DESC, created_at DESC
                    """,
                        (session_id, entity_type),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM session_entities 
                        WHERE session_id=?
                        ORDER BY confidence_score DESC, created_at DESC
                    """,
                        (session_id,),
                    )

                entities = []
                for row in cursor:
                    entity_dict = dict(row)
                    entity_dict["attributes"] = (
                        json.loads(entity_dict["attributes"])
                        if entity_dict["attributes"]
                        else {}
                    )
                    entities.append(entity_dict)
                return entities

        try:
            entities = await asyncio.get_event_loop().run_in_executor(None, _get_sync)
            session_mem_logger.debug(f"Retrieved {len(entities)} session entities.")
            return entities
        except Exception as e:
            session_mem_logger.error("Failed to get session entities.", exception=e)
            raise MemoryManagerError("Failed to get session entities.", cause=e)

    # --- Context Window Management ---
    @detailed_log_function(LogCategory.DATABASE)
    async def add_context_window_entry(
        self,
        session_id: str,
        entry_type: str,
        content: Any,
        token_count: Optional[int] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        memory_type: MemoryType = MemoryType.CONTEXT_WINDOW,
    ) -> MemoryEntry:
        context_mem_logger.info(
            "Adding context window entry.",
            parameters={"session": session_id, "type": entry_type},
        )
        self._record_op("add_context_window_entry")

        entry_id = str(uuid.uuid4())
        entry = MemoryEntry(
            id=entry_id,
            memory_type=memory_type,
            key=entry_type,
            value=content,
            session_id=session_id,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            importance_score=importance,
        )
        content_json = json.dumps(entry.value, default=str)
        metadata_json = json.dumps(entry.metadata, default=str)
        created_at_iso = entry.created_at.isoformat()

        # Estimate token_count if not provided (very rough estimate)
        if token_count is None:
            token_count = len(content_json.split()) // 2

        def _add_sync():
            with self._lock, self._get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO context_window_entries
                        (entry_id, session_id, entry_type, content, token_count, importance_score, memory_type, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entry.id,
                        entry.session_id,
                        entry_type,
                        content_json,
                        token_count,
                        entry.importance_score,
                        entry.memory_type.value,
                        created_at_iso,
                        metadata_json,
                    ),
                )
                conn.commit()

        try:
            await asyncio.get_event_loop().run_in_executor(None, _add_sync)
            context_mem_logger.debug(
                "Context window entry added.", parameters={"id": entry.id}
            )
            # After adding, check if pruning is needed
            asyncio.create_task(self.prune_context_window(session_id))
            return entry
        except Exception as e:
            context_mem_logger.error("Failed to add context window entry.", exception=e)
            raise MemoryManagerError("Failed to add context window entry.", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def get_context_window(
        self, session_id: str, max_tokens: Optional[int] = None
    ) -> List[MemoryEntry]:
        context_mem_logger.debug(
            "Retrieving context window.",
            parameters={"session": session_id, "max_tokens": max_tokens},
        )
        self._record_op("get_context_window")
        target_tokens = max_tokens or self.max_context_tokens

        def _get_sync():
            entries: List[MemoryEntry] = []
            current_tokens = 0
            with self._lock, self._get_db_connection() as conn:
                conn.row_factory = sqlite3.Row
                # Fetch entries, most recent and most important first
                cursor = conn.execute(
                    """
                    SELECT * FROM context_window_entries
                    WHERE session_id = ?
                    ORDER BY created_at DESC 
                """,
                    (session_id,),
                )

                all_rows = cursor.fetchall()

                # Sort by importance then recency for prompt construction
                sorted_rows = sorted(
                    all_rows,
                    key=lambda r: (r["importance_score"], r["created_at"]),
                    reverse=True,
                )

                for row in sorted_rows:
                    row_dict = dict(row)
                    entry_tokens = row_dict.get(
                        "token_count", len(row_dict["content"].split()) // 2
                    )
                    if current_tokens + entry_tokens <= target_tokens:
                        entry = MemoryEntry(
                            id=row_dict["entry_id"],
                            memory_type=MemoryType(row_dict["memory_type"]),
                            key=row_dict["entry_type"],
                            value=json.loads(row_dict["content"]),
                            session_id=row_dict["session_id"],
                            metadata=json.loads(row_dict["metadata"]) if row_dict["metadata"] else {},
                            created_at=datetime.fromisoformat(row_dict["created_at"]),
                            updated_at=datetime.fromisoformat(row_dict["created_at"]),
                            importance_score=row_dict["importance_score"],
                        )
                        entries.append(entry)
                        current_tokens += entry_tokens
                    else:
                        break
            return entries

        try:
            retrieved_entries = await asyncio.get_event_loop().run_in_executor(
                None, _get_sync
            )
            retrieved_entries.sort(key=lambda x: x.created_at)
            context_mem_logger.debug(
                f"Retrieved {len(retrieved_entries)} entries for context window."
            )
            return retrieved_entries
        except Exception as e:
            context_mem_logger.error("Failed to get context window.", exception=e)
            raise MemoryManagerError("Failed to get context window.", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def prune_context_window(
        self, session_id: str, target_tokens: Optional[int] = None
    ):
        """Prunes the context window for a session to stay within token limits."""
        context_mem_logger.info(
            "Pruning context window.",
            parameters={"session": session_id, "target_tokens": target_tokens},
        )
        self._record_op("prune_context_window")
        effective_target_tokens = target_tokens or self.max_context_tokens

        def _prune_sync():
            with self._lock, self._get_db_connection() as conn:
                conn.row_factory = sqlite3.Row
                # Get all entries, sorted by importance (desc) and then recency (desc) to keep important/recent ones
                cursor = conn.execute(
                    """
                    SELECT entry_id, token_count, importance_score, created_at FROM context_window_entries
                    WHERE session_id = ?
                    ORDER BY importance_score DESC, created_at DESC
                """,
                    (session_id,),
                )

                entries_to_keep_ids: List[str] = []
                current_token_sum = 0
                all_entries = cursor.fetchall()

                for row in all_entries:
                    entry_token_count = (
                        row["token_count"] if row["token_count"] is not None else 50
                    )
                    if current_token_sum + entry_token_count <= effective_target_tokens:
                        entries_to_keep_ids.append(row["entry_id"])
                        current_token_sum += entry_token_count
                    else:
                        pass

                if len(entries_to_keep_ids) < len(all_entries):
                    # SQL to delete entries NOT IN the keep list
                    placeholders = ",".join("?" for _ in entries_to_keep_ids)
                    if entries_to_keep_ids:
                        delete_cursor = conn.execute(
                            f"DELETE FROM context_window_entries WHERE session_id = ? AND entry_id NOT IN ({placeholders})",
                            [session_id] + entries_to_keep_ids,
                        )
                    else:
                        delete_cursor = conn.execute(
                            "DELETE FROM context_window_entries WHERE session_id = ?",
                            (session_id,),
                        )
                    conn.commit()
                    context_mem_logger.info(
                        f"Context window pruned.",
                        parameters={
                            "session": session_id,
                            "deleted_count": delete_cursor.rowcount,
                            "kept_count": len(entries_to_keep_ids),
                            "current_tokens": current_token_sum,
                        },
                    )
                else:
                    context_mem_logger.debug(
                        "No pruning needed for context window.",
                        parameters={"session": session_id},
                    )

        try:
            await asyncio.get_event_loop().run_in_executor(None, _prune_sync)
        except Exception as e:
            context_mem_logger.error("Failed to prune context window.", exception=e)
            raise MemoryManagerError("Failed to prune context window.", cause=e)

    # --- Agent Decisions Log ---
    @detailed_log_function(LogCategory.DATABASE)
    async def log_agent_decision(
        self,
        agent_name: str,
        session_id: Optional[str],
        input_summary: str,
        decision_details: Dict[str, Any],
        context_used: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
    ) -> str:
        agent_mem_logger.info(
            "Logging agent decision.",
            parameters={"agent": agent_name, "session": session_id},
        )
        self._record_op("log_agent_decision")

        decision_id = str(uuid.uuid4())
        decision_json = json.dumps(decision_details, default=str)
        context_json = json.dumps(context_used, default=str) if context_used else "{}"
        tags_json = json.dumps(tags, default=str) if tags else "[]"
        created_at_iso = datetime.now(timezone.utc).isoformat()

        def _log_sync():
            with self._lock, self._get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO agent_decisions_log
                        (decision_id, agent_name, session_id, input_summary, decision_details, 
                         context_used, confidence_score, tags, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        decision_id,
                        agent_name,
                        session_id,
                        input_summary,
                        decision_json,
                        context_json,
                        confidence,
                        tags_json,
                        created_at_iso,
                    ),
                )
                conn.commit()

        try:
            await asyncio.get_event_loop().run_in_executor(None, _log_sync)
            agent_mem_logger.debug(
                "Agent decision logged.", parameters={"id": decision_id}
            )
            return decision_id
        except Exception as e:
            agent_mem_logger.error("Failed to log agent decision.", exception=e)
            raise MemoryManagerError("Failed to log agent decision.", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def get_agent_decisions(
        self, agent_name: str, session_id: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        agent_mem_logger.debug(
            "Getting agent decisions.",
            parameters={"agent": agent_name, "session": session_id, "limit": limit},
        )
        self._record_op("get_agent_decisions")

        def _get_sync():
            with self._lock, self._get_db_connection() as conn:
                conn.row_factory = sqlite3.Row
                if session_id:
                    cursor = conn.execute(
                        """
                        SELECT * FROM agent_decisions_log
                        WHERE agent_name = ? AND session_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    """,
                        (agent_name, session_id, limit),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM agent_decisions_log
                        WHERE agent_name = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    """,
                        (agent_name, limit),
                    )

                decisions = []
                for row in cursor:
                    decision_dict = dict(row)
                    decision_dict["decision_details"] = json.loads(
                        decision_dict["decision_details"]
                    )
                    decision_dict["context_used"] = (
                        json.loads(decision_dict["context_used"])
                        if decision_dict["context_used"]
                        else {}
                    )
                    decision_dict["tags"] = (
                        json.loads(decision_dict["tags"])
                        if decision_dict["tags"]
                        else []
                    )
                    decisions.append(decision_dict)
                return decisions

        try:
            decisions = await asyncio.get_event_loop().run_in_executor(None, _get_sync)
            agent_mem_logger.debug(f"Retrieved {len(decisions)} agent decisions.")
            return decisions
        except Exception as e:
            agent_mem_logger.error("Failed to get agent decisions.", exception=e)
            raise MemoryManagerError("Failed to get agent decisions.", cause=e)

    # --- Learning Stats ---
    @detailed_log_function(LogCategory.DATABASE)
    async def get_tag_learning_stats_async(
        self, tag_text: str
    ) -> Optional[Dict[str, Any]]:
        self._record_op("get_tag_learning_stats")

        def _get_sync():
            with self._lock, self._get_db_connection() as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM tag_learning_stats WHERE tag_text = ?", (tag_text,)
                ).fetchone()
                return dict(row) if row else None

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _get_sync)
        except Exception as e:
            umm_logger.error(
                "Failed to get tag learning stats.",
                parameters={"tag": tag_text},
                exception=e,
            )
            return None

    @detailed_log_function(LogCategory.DATABASE)
    async def update_tag_learning_stats_async(
        self,
        tag_text: str,
        correct_increment: int = 0,
        incorrect_increment: int = 0,
        suggested_increment: int = 0,
    ) -> None:
        self._record_op("update_tag_learning_stats")
        now_iso = datetime.now(timezone.utc).isoformat()

        def _update_sync():
            with self._lock, self._get_db_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO tag_learning_stats (tag_text, correct_count, incorrect_count, suggested_count, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(tag_text) DO UPDATE SET
                        correct_count = correct_count + excluded.correct_count,
                        incorrect_count = incorrect_count + excluded.incorrect_count,
                        suggested_count = suggested_count + excluded.suggested_count,
                        last_updated = excluded.last_updated;
                """,
                    (
                        tag_text,
                        correct_increment,
                        incorrect_increment,
                        suggested_increment,
                        now_iso,
                    ),
                )
                conn.commit()

        try:
            await asyncio.get_event_loop().run_in_executor(None, _update_sync)
            umm_logger.trace(
                "Tag learning stats updated.", parameters={"tag": tag_text}
            )
        except Exception as e:
            umm_logger.error(
                "Failed to update tag learning stats.",
                parameters={"tag": tag_text},
                exception=e,
            )

    @detailed_log_function(LogCategory.DATABASE)
    async def get_pattern_stats_async(
        self, pattern_regex: str
    ) -> Optional[Dict[str, Any]]:
        self._record_op("get_pattern_stats")
        pattern_hash = hashlib.md5(pattern_regex.encode()).hexdigest()

        def _get_sync():
            with self._lock, self._get_db_connection() as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM pattern_learning_stats WHERE pattern_hash = ?",
                    (pattern_hash,),
                ).fetchone()
                return dict(row) if row else None

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _get_sync)
        except Exception as e:
            umm_logger.error(
                "Failed to get pattern stats.",
                parameters={"pattern": pattern_regex},
                exception=e,
            )
            return None

    @detailed_log_function(LogCategory.DATABASE)
    async def update_pattern_stats_async(
        self,
        pattern_regex: str,
        effectiveness_score: Optional[float] = None,
        usage_increment: int = 1,
    ) -> None:
        self._record_op("update_pattern_stats")
        pattern_hash = hashlib.md5(pattern_regex.encode()).hexdigest()
        now_iso = datetime.now(timezone.utc).isoformat()

        def _update_sync():
            with self._lock, self._get_db_connection() as conn:
                if effectiveness_score is not None:
                    conn.execute(
                        """
                        INSERT INTO pattern_learning_stats 
                            (pattern_hash, pattern_regex, effectiveness_score, usage_count, last_updated)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(pattern_hash) DO UPDATE SET
                            effectiveness_score = excluded.effectiveness_score,
                            usage_count = usage_count + excluded.usage_count,
                            last_updated = excluded.last_updated;
                    """,
                        (
                            pattern_hash,
                            pattern_regex,
                            effectiveness_score,
                            usage_increment,
                            now_iso,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO pattern_learning_stats 
                            (pattern_hash, pattern_regex, usage_count, last_updated)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(pattern_hash) DO UPDATE SET
                            usage_count = usage_count + excluded.usage_count,
                            last_updated = excluded.last_updated;
                    """,
                        (pattern_hash, pattern_regex, usage_increment, now_iso),
                    )
                conn.commit()

        try:
            await asyncio.get_event_loop().run_in_executor(None, _update_sync)
            umm_logger.trace(
                "Pattern stats updated.", parameters={"pattern": pattern_regex}
            )
        except Exception as e:
            umm_logger.error(
                "Failed to update pattern stats.",
                parameters={"pattern": pattern_regex},
                exception=e,
            )

    # --- Statistics and Health ---
    @detailed_log_function(LogCategory.DATABASE)
    async def get_statistics(self) -> Dict[str, Any]:
        umm_logger.debug("Fetching UnifiedMemoryManager statistics.")
        self._record_op("get_statistics")

        def _get_stats_sync():
            stats: Dict[str, Any] = {
                "db_path": str(self.db_path),
                "max_context_tokens": self.max_context_tokens,
                "operation_counts": self._operation_counts.copy(),
                "last_operation_time": (
                    self._last_operation_time.isoformat()
                    if self._last_operation_time
                    else None
                ),
            }
            try:
                with self._get_db_connection() as conn:
                    stats["agent_memory_entries"] = conn.execute(
                        "SELECT COUNT(*) FROM agent_memory"
                    ).fetchone()[0]
                    stats["session_entities_entries"] = conn.execute(
                        "SELECT COUNT(*) FROM session_entities"
                    ).fetchone()[0]
                    stats["session_observations_entries"] = conn.execute(
                        "SELECT COUNT(*) FROM session_observations"
                    ).fetchone()[0]
                    stats["session_relationships_entries"] = conn.execute(
                        "SELECT COUNT(*) FROM session_relationships"
                    ).fetchone()[0]
                    stats["context_window_total_entries"] = conn.execute(
                        "SELECT COUNT(*) FROM context_window_entries"
                    ).fetchone()[0]
                    stats["agent_decisions_log_entries"] = conn.execute(
                        "SELECT COUNT(*) FROM agent_decisions_log"
                    ).fetchone()[0]
                    stats["tag_learning_stats_entries"] = conn.execute(
                        "SELECT COUNT(*) FROM tag_learning_stats"
                    ).fetchone()[0]
                    stats["pattern_learning_stats_entries"] = conn.execute(
                        "SELECT COUNT(*) FROM pattern_learning_stats"
                    ).fetchone()[0]

                    # Example: Total tokens in context windows (approx)
                    token_sum_result = conn.execute(
                        "SELECT SUM(token_count) FROM context_window_entries WHERE token_count IS NOT NULL"
                    ).fetchone()
                    stats["context_window_total_tokens_approx"] = (
                        token_sum_result[0]
                        if token_sum_result and token_sum_result[0] is not None
                        else 0
                    )

                    db_size_bytes = (
                        self.db_path.stat().st_size if self.db_path.exists() else 0
                    )
                    stats["db_size_mb"] = round(db_size_bytes / (1024 * 1024), 2)

            except Exception as e:
                umm_logger.error("Failed to get detailed DB statistics.", exception=e)
                stats["db_error"] = str(e)
            return stats

        try:
            full_stats = await asyncio.get_event_loop().run_in_executor(
                None, _get_stats_sync
            )
            umm_logger.info("UnifiedMemoryManager statistics retrieved.")
            return full_stats
        except Exception as e:
            umm_logger.error("Failed to retrieve statistics.", exception=e)
            return {"error": str(e)}

    async def get_service_status(self) -> Dict[str, Any]:
        umm_logger.debug(
            "Performing UnifiedMemoryManager health check for service status."
        )
        is_healthy = self._initialized
        db_connect_ok = False
        try:
            # Test DB connection
            conn = self._get_db_connection()
            conn.close()
            db_connect_ok = True
        except Exception:
            db_connect_ok = False
            is_healthy = False

        status_report = {
            "status": (
                "healthy"
                if is_healthy and db_connect_ok
                else ("degraded" if self._initialized else "error")
            ),
            "initialized": self._initialized,
            "database_path": str(self.db_path),
            "database_connection": "ok" if db_connect_ok else "error",
            "max_context_tokens": self.max_context_tokens,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if is_healthy and db_connect_ok:
            stats = await self.get_statistics()
            status_report.update(
                {
                    "agent_memory_count": stats.get("agent_memory_entries", "N/A"),
                    "session_knowledge_count": stats.get(
                        "session_entities_entries", "N/A"
                    ),
                    "context_entry_count": stats.get(
                        "context_window_total_entries", "N/A"
                    ),
                }
            )
        umm_logger.info(
            "UnifiedMemoryManager health check complete.", parameters=status_report
        )
        return status_report

    async def close(self):
        umm_logger.info("Closing UnifiedMemoryManager.")
        # SQLite connections are typically opened and closed per operation or managed by context managers.
        # No explicit pool to close here unless one was implemented.
        self._initialized = False
        umm_logger.info("UnifiedMemoryManager closed.")


# Factory for service container
def create_unified_memory_manager(
    service_config: Optional[UnifiedMemoryManagerConfig] = None,
) -> UnifiedMemoryManager:
    cfg = service_config or UnifiedMemoryManagerConfig()

    return UnifiedMemoryManager(
        db_path_str=str(cfg.db_path),
        max_context_tokens_config=int(cfg.max_context_tokens),
        service_config=cfg,
    )
