"""
Claude Memory Store - Persistent Memory for Claude Code Sessions

This module provides a dedicated SQLite database for storing Claude's memory
across sessions, including entities, relationships, observations, and session context.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import threading
import hashlib

logger = logging.getLogger(__name__)


class ClaudeMemoryStore:
    """
    Persistent memory store for Claude Code sessions
    Stores entities, relationships, observations, and session context
    """
    
    def __init__(self, db_path: str = "./storage/databases/claude_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._init_database()
        
        logger.info(f"Claude Memory Store initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Entities table
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    entity_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                );
                
                -- Observations table
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    importance_score REAL DEFAULT 0.5,
                    source TEXT DEFAULT 'claude',
                    FOREIGN KEY (entity_name) REFERENCES entities (name) ON DELETE CASCADE
                );
                
                -- Relations table
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_entity TEXT NOT NULL,
                    to_entity TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    strength REAL DEFAULT 1.0,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (from_entity) REFERENCES entities (name) ON DELETE CASCADE,
                    FOREIGN KEY (to_entity) REFERENCES entities (name) ON DELETE CASCADE,
                    UNIQUE(from_entity, to_entity, relation_type)
                );
                
                -- Sessions table for context tracking
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    session_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context_data TEXT DEFAULT '{}',
                    active BOOLEAN DEFAULT 1
                );
                
                -- Session entities for tracking what entities were discussed in sessions
                CREATE TABLE IF NOT EXISTS session_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    mentioned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    relevance_score REAL DEFAULT 1.0,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
                    FOREIGN KEY (entity_name) REFERENCES entities (name) ON DELETE CASCADE,
                    UNIQUE(session_id, entity_name)
                );
                
                -- Knowledge facts for storing key insights
                CREATE TABLE IF NOT EXISTS knowledge_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_hash TEXT UNIQUE NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    verified BOOLEAN DEFAULT 0,
                    source_entity TEXT,
                    FOREIGN KEY (source_entity) REFERENCES entities (name) ON DELETE SET NULL
                );
                
                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (entity_type);
                CREATE INDEX IF NOT EXISTS idx_entities_name ON entities (name);
                CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations (entity_name);
                CREATE INDEX IF NOT EXISTS idx_observations_importance ON observations (importance_score);
                CREATE INDEX IF NOT EXISTS idx_relations_from ON relations (from_entity);
                CREATE INDEX IF NOT EXISTS idx_relations_to ON relations (to_entity);
                CREATE INDEX IF NOT EXISTS idx_relations_type ON relations (relation_type);
                CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions (active);
                CREATE INDEX IF NOT EXISTS idx_session_entities_session ON session_entities (session_id);
                CREATE INDEX IF NOT EXISTS idx_knowledge_facts_subject ON knowledge_facts (subject);
                CREATE INDEX IF NOT EXISTS idx_knowledge_facts_hash ON knowledge_facts (fact_hash);
            """)
            conn.commit()
    
    def create_session(self, session_name: str = None) -> str:
        """Create a new session and return session ID"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, session_name, context_data)
                VALUES (?, ?, ?)
            """, (session_id, session_name or f"Session {session_id}", json.dumps({})))
            conn.commit()
        
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def store_entity(self, name: str, entity_type: str, metadata: Dict[str, Any] = None) -> bool:
        """Store or update an entity"""
        try:
            with self._lock, sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO entities (name, entity_type, updated_at, metadata)
                    VALUES (?, ?, CURRENT_TIMESTAMP, ?)
                """, (name, entity_type, json.dumps(metadata or {})))
                conn.commit()
            
            logger.debug(f"Stored entity: {name} ({entity_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store entity {name}: {e}")
            return False
    
    def add_observation(self, entity_name: str, content: str, 
                       importance_score: float = 0.5, source: str = "claude") -> bool:
        """Add an observation to an entity"""
        try:
            with self._lock, sqlite3.connect(self.db_path) as conn:
                # Ensure entity exists
                conn.execute("""
                    INSERT OR IGNORE INTO entities (name, entity_type)
                    VALUES (?, 'unknown')
                """, (entity_name,))
                
                # Add observation
                conn.execute("""
                    INSERT INTO observations (entity_name, content, importance_score, source)
                    VALUES (?, ?, ?, ?)
                """, (entity_name, content, importance_score, source))
                conn.commit()
            
            logger.debug(f"Added observation to {entity_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add observation to {entity_name}: {e}")
            return False
    
    def create_relation(self, from_entity: str, to_entity: str, relation_type: str,
                       strength: float = 1.0, metadata: Dict[str, Any] = None) -> bool:
        """Create a relationship between entities"""
        try:
            with self._lock, sqlite3.connect(self.db_path) as conn:
                # Ensure both entities exist
                for entity in [from_entity, to_entity]:
                    conn.execute("""
                        INSERT OR IGNORE INTO entities (name, entity_type)
                        VALUES (?, 'unknown')
                    """, (entity,))
                
                # Create relation
                conn.execute("""
                    INSERT OR REPLACE INTO relations 
                    (from_entity, to_entity, relation_type, strength, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (from_entity, to_entity, relation_type, strength, json.dumps(metadata or {})))
                conn.commit()
            
            logger.debug(f"Created relation: {from_entity} --{relation_type}--> {to_entity}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create relation {from_entity} -> {to_entity}: {e}")
            return False
    
    def store_knowledge_fact(self, subject: str, predicate: str, object_val: str,
                           confidence: float = 1.0, source_entity: str = None) -> bool:
        """Store a knowledge fact (subject-predicate-object triple)"""
        try:
            # Create hash for deduplication
            fact_text = f"{subject}|{predicate}|{object_val}"
            fact_hash = hashlib.md5(fact_text.encode()).hexdigest()
            
            with self._lock, sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO knowledge_facts 
                    (fact_hash, subject, predicate, object, confidence, source_entity)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (fact_hash, subject, predicate, object_val, confidence, source_entity))
                conn.commit()
            
            logger.debug(f"Stored fact: {subject} {predicate} {object_val}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store knowledge fact: {e}")
            return False
    
    def link_session_entity(self, session_id: str, entity_name: str, 
                           relevance_score: float = 1.0) -> bool:
        """Link an entity to a session"""
        try:
            with self._lock, sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO session_entities 
                    (session_id, entity_name, mentioned_at, relevance_score)
                    VALUES (?, ?, CURRENT_TIMESTAMP, ?)
                """, (session_id, entity_name, relevance_score))
                conn.commit()
            
            logger.debug(f"Linked {entity_name} to session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to link entity to session: {e}")
            return False
    
    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """Get an entity with its observations"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Get entity
                cursor = conn.execute("""
                    SELECT * FROM entities WHERE name = ?
                """, (name,))
                entity_row = cursor.fetchone()
                
                if not entity_row:
                    return None
                
                # Get observations
                cursor = conn.execute("""
                    SELECT content, importance_score, created_at, source
                    FROM observations 
                    WHERE entity_name = ?
                    ORDER BY importance_score DESC, created_at DESC
                """, (name,))
                observations = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'name': entity_row['name'],
                    'entity_type': entity_row['entity_type'],
                    'created_at': entity_row['created_at'],
                    'updated_at': entity_row['updated_at'],
                    'metadata': json.loads(entity_row['metadata']),
                    'observations': observations
                }
                
        except Exception as e:
            logger.error(f"Failed to get entity {name}: {e}")
            return None
    
    def search_entities(self, query: str, entity_type: str = None, 
                       limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities by name or observations"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Build query
                sql = """
                    SELECT DISTINCT e.name, e.entity_type, e.created_at, e.updated_at,
                           COUNT(o.id) as observation_count
                    FROM entities e
                    LEFT JOIN observations o ON e.name = o.entity_name
                    WHERE (e.name LIKE ? OR o.content LIKE ?)
                """
                params = [f"%{query}%", f"%{query}%"]
                
                if entity_type:
                    sql += " AND e.entity_type = ?"
                    params.append(entity_type)
                
                sql += """
                    GROUP BY e.name, e.entity_type, e.created_at, e.updated_at
                    ORDER BY observation_count DESC, e.updated_at DESC
                    LIMIT ?
                """
                params.append(limit)
                
                cursor = conn.execute(sql, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to search entities: {e}")
            return []
    
    def get_related_entities(self, entity_name: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get entities related to the given entity"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Get direct relationships
                cursor = conn.execute("""
                    SELECT 
                        CASE 
                            WHEN from_entity = ? THEN to_entity 
                            ELSE from_entity 
                        END as related_entity,
                        relation_type,
                        strength,
                        CASE 
                            WHEN from_entity = ? THEN 'outgoing' 
                            ELSE 'incoming' 
                        END as direction
                    FROM relations 
                    WHERE from_entity = ? OR to_entity = ?
                    ORDER BY strength DESC
                """, (entity_name, entity_name, entity_name, entity_name))
                
                related = []
                for row in cursor.fetchall():
                    related.append({
                        'entity_name': row['related_entity'],
                        'relation_type': row['relation_type'],
                        'direction': row['direction'],
                        'strength': row['strength'],
                        'depth': 1
                    })
                
                return related
                
        except Exception as e:
            logger.error(f"Failed to get related entities for {entity_name}: {e}")
            return []
    
    def get_knowledge_facts(self, subject: str = None, predicate: str = None,
                          limit: int = 50) -> List[Dict[str, Any]]:
        """Get knowledge facts, optionally filtered"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                sql = "SELECT * FROM knowledge_facts WHERE 1=1"
                params = []
                
                if subject:
                    sql += " AND subject LIKE ?"
                    params.append(f"%{subject}%")
                
                if predicate:
                    sql += " AND predicate LIKE ?"
                    params.append(f"%{predicate}%")
                
                sql += " ORDER BY confidence DESC, created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(sql, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get knowledge facts: {e}")
            return []
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get session context and related entities"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Get session info
                cursor = conn.execute("""
                    SELECT * FROM sessions WHERE session_id = ?
                """, (session_id,))
                session_row = cursor.fetchone()
                
                if not session_row:
                    return {}
                
                # Get session entities
                cursor = conn.execute("""
                    SELECT se.entity_name, se.relevance_score, se.mentioned_at,
                           e.entity_type
                    FROM session_entities se
                    JOIN entities e ON se.entity_name = e.name
                    WHERE se.session_id = ?
                    ORDER BY se.relevance_score DESC, se.mentioned_at DESC
                """, (session_id,))
                entities = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'session_id': session_row['session_id'],
                    'session_name': session_row['session_name'],
                    'created_at': session_row['created_at'],
                    'updated_at': session_row['updated_at'],
                    'context_data': json.loads(session_row['context_data']),
                    'entities': entities
                }
                
        except Exception as e:
            logger.error(f"Failed to get session context: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Count entities
                cursor = conn.execute("SELECT COUNT(*) FROM entities")
                stats['total_entities'] = cursor.fetchone()[0]
                
                # Count observations
                cursor = conn.execute("SELECT COUNT(*) FROM observations")
                stats['total_observations'] = cursor.fetchone()[0]
                
                # Count relations
                cursor = conn.execute("SELECT COUNT(*) FROM relations")
                stats['total_relations'] = cursor.fetchone()[0]
                
                # Count sessions
                cursor = conn.execute("SELECT COUNT(*) FROM sessions")
                stats['total_sessions'] = cursor.fetchone()[0]
                
                # Count knowledge facts
                cursor = conn.execute("SELECT COUNT(*) FROM knowledge_facts")
                stats['total_knowledge_facts'] = cursor.fetchone()[0]
                
                # Entity types
                cursor = conn.execute("""
                    SELECT entity_type, COUNT(*) as count 
                    FROM entities 
                    GROUP BY entity_type 
                    ORDER BY count DESC
                """)
                stats['entity_types'] = dict(cursor.fetchall())
                
                # Database size
                stats['db_path'] = str(self.db_path)
                try:
                    stats['db_size_bytes'] = self.db_path.stat().st_size
                except:
                    stats['db_size_bytes'] = 0
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def export_memory_graph(self) -> Dict[str, Any]:
        """Export the complete memory as a graph structure"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Get all entities with observations
                entities = []
                cursor = conn.execute("""
                    SELECT e.*, 
                           GROUP_CONCAT(o.content, '|') as observations,
                           COUNT(o.id) as observation_count
                    FROM entities e
                    LEFT JOIN observations o ON e.name = o.entity_name
                    GROUP BY e.name
                """)
                
                for row in cursor.fetchall():
                    entity = dict(row)
                    entity['metadata'] = json.loads(entity['metadata'])
                    if entity['observations']:
                        entity['observations'] = entity['observations'].split('|')
                    else:
                        entity['observations'] = []
                    entities.append(entity)
                
                # Get all relations
                cursor = conn.execute("SELECT * FROM relations")
                relations = [dict(row) for row in cursor.fetchall()]
                
                # Get knowledge facts
                cursor = conn.execute("SELECT * FROM knowledge_facts ORDER BY confidence DESC")
                facts = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'entities': entities,
                    'relations': relations,
                    'knowledge_facts': facts,
                    'exported_at': datetime.now().isoformat(),
                    'statistics': self.get_statistics()
                }
                
        except Exception as e:
            logger.error(f"Failed to export memory graph: {e}")
            return {}
    
    def close(self):
        """Close the memory store"""
        logger.info("Claude Memory Store closed")


# Global instance for easy access
_claude_memory = None

def get_claude_memory() -> ClaudeMemoryStore:
    """Get the global Claude memory instance"""
    global _claude_memory
    if _claude_memory is None:
        _claude_memory = ClaudeMemoryStore()
    return _claude_memory
