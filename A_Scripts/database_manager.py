"""
Database Manager for Legal AI System GUI
Handles database interactions for violations, memory, and knowledge graph data
"""

import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class ViolationRecord:
    """Violation record structure"""
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
    comments: Optional[str] = None

@dataclass
class MemoryRecord:
    """Memory record structure"""
    id: str
    memory_type: str
    content: str
    confidence: float
    source_document: str
    created_time: datetime
    last_accessed: datetime
    access_count: int
    tags: Optional[str] = None  # JSON string
    metadata: Optional[str] = None  # JSON string

@dataclass
class GraphNode:
    """Knowledge graph node structure"""
    id: str
    label: str
    node_type: str
    properties: str  # JSON string
    created_time: datetime
    updated_time: datetime

@dataclass
class GraphEdge:
    """Knowledge graph edge structure"""
    id: str
    from_node: str
    to_node: str
    relationship_type: str
    weight: float
    properties: str  # JSON string
    created_time: datetime

class DatabaseManager:
    """Manages SQLite database operations for the GUI"""
    
    def __init__(self, db_path: str = "legal_ai_gui.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with required tables"""
        with self._get_connection() as conn:
            # Violations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS violations (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    detected_time TIMESTAMP NOT NULL,
                    reviewed_by TEXT,
                    review_time TIMESTAMP,
                    comments TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Memory table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_document TEXT NOT NULL,
                    created_time TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    tags TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Knowledge graph nodes table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    properties TEXT,
                    created_time TIMESTAMP NOT NULL,
                    updated_time TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Knowledge graph edges table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_edges (
                    id TEXT PRIMARY KEY,
                    from_node TEXT NOT NULL,
                    to_node TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    weight REAL NOT NULL,
                    properties TEXT,
                    created_time TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (from_node) REFERENCES graph_nodes (id),
                    FOREIGN KEY (to_node) REFERENCES graph_nodes (id)
                )
            """)
            
            # Documents table for tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    upload_time TIMESTAMP NOT NULL,
                    processing_status TEXT NOT NULL,
                    processing_options TEXT,
                    results TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # System logs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_status ON violations(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_severity ON violations(severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_document ON violations(document_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_document ON memory_entries(source_document)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs(level)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_component ON system_logs(component)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            conn.close()
    
    # Violation management methods
    def save_violation(self, violation: ViolationRecord) -> bool:
        """Save a violation record to the database"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO violations (
                        id, document_id, violation_type, severity, status,
                        description, confidence, detected_time, reviewed_by,
                        review_time, comments
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    violation.id, violation.document_id, violation.violation_type,
                    violation.severity, violation.status, violation.description,
                    violation.confidence, violation.detected_time, violation.reviewed_by,
                    violation.review_time, violation.comments
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save violation: {e}")
            return False
    
    def get_violations(self, filters: Optional[Dict[str, Any]] = None) -> List[ViolationRecord]:
        """Retrieve violations with optional filters"""
        try:
            query = "SELECT * FROM violations WHERE 1=1"
            params = []
            
            if filters:
                if "status" in filters and filters["status"]:
                    placeholders = ",".join(["?" for _ in filters["status"]])
                    query += f" AND status IN ({placeholders})"
                    params.extend(filters["status"])
                
                if "severity" in filters and filters["severity"]:
                    placeholders = ",".join(["?" for _ in filters["severity"]])
                    query += f" AND severity IN ({placeholders})"
                    params.extend(filters["severity"])
                
                if "violation_type" in filters and filters["violation_type"]:
                    placeholders = ",".join(["?" for _ in filters["violation_type"]])
                    query += f" AND violation_type IN ({placeholders})"
                    params.extend(filters["violation_type"])
                
                if "min_confidence" in filters:
                    query += " AND confidence >= ?"
                    params.append(filters["min_confidence"])
                
                if "start_date" in filters:
                    query += " AND detected_time >= ?"
                    params.append(filters["start_date"])
                
                if "end_date" in filters:
                    query += " AND detected_time <= ?"
                    params.append(filters["end_date"])
            
            query += " ORDER BY detected_time DESC"
            
            with self._get_connection() as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                violations = []
                for row in rows:
                    violation = ViolationRecord(
                        id=row["id"],
                        document_id=row["document_id"],
                        violation_type=row["violation_type"],
                        severity=row["severity"],
                        status=row["status"],
                        description=row["description"],
                        confidence=row["confidence"],
                        detected_time=datetime.fromisoformat(row["detected_time"]) if isinstance(row["detected_time"], str) else row["detected_time"],
                        reviewed_by=row["reviewed_by"],
                        review_time=datetime.fromisoformat(row["review_time"]) if row["review_time"] and isinstance(row["review_time"], str) else row["review_time"],
                        comments=row["comments"]
                    )
                    violations.append(violation)
                
                return violations
        except Exception as e:
            logger.error(f"Failed to retrieve violations: {e}")
            return []
    
    def update_violation_status(self, violation_id: str, status: str, reviewed_by: str, comments: Optional[str] = None) -> bool:
        """Update violation status and review information"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE violations 
                    SET status = ?, reviewed_by = ?, review_time = ?, comments = ?
                    WHERE id = ?
                """, (status, reviewed_by, datetime.now(), comments, violation_id))
                conn.commit()
                return conn.total_changes > 0
        except Exception as e:
            logger.error(f"Failed to update violation status: {e}")
            return False
    
    # Memory management methods
    def save_memory_entry(self, memory: MemoryRecord) -> bool:
        """Save a memory entry to the database"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memory_entries (
                        id, memory_type, content, confidence, source_document,
                        created_time, last_accessed, access_count, tags, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id, memory.memory_type, memory.content, memory.confidence,
                    memory.source_document, memory.created_time, memory.last_accessed,
                    memory.access_count, memory.tags, memory.metadata
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save memory entry: {e}")
            return False
    
    def get_memory_entries(self, filters: Optional[Dict[str, Any]] = None) -> List[MemoryRecord]:
        """Retrieve memory entries with optional filters"""
        try:
            query = "SELECT * FROM memory_entries WHERE 1=1"
            params = []
            
            if filters:
                if "memory_type" in filters and filters["memory_type"]:
                    placeholders = ",".join(["?" for _ in filters["memory_type"]])
                    query += f" AND memory_type IN ({placeholders})"
                    params.extend(filters["memory_type"])
                
                if "min_confidence" in filters:
                    query += " AND confidence >= ?"
                    params.append(filters["min_confidence"])
                
                if "source_document" in filters:
                    query += " AND source_document = ?"
                    params.append(filters["source_document"])
                
                if "search_content" in filters:
                    query += " AND content LIKE ?"
                    params.append(f"%{filters['search_content']}%")
            
            query += " ORDER BY last_accessed DESC"
            
            with self._get_connection() as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                memories = []
                for row in rows:
                    memory = MemoryRecord(
                        id=row["id"],
                        memory_type=row["memory_type"],
                        content=row["content"],
                        confidence=row["confidence"],
                        source_document=row["source_document"],
                        created_time=datetime.fromisoformat(row["created_time"]) if isinstance(row["created_time"], str) else row["created_time"],
                        last_accessed=datetime.fromisoformat(row["last_accessed"]) if isinstance(row["last_accessed"], str) else row["last_accessed"],
                        access_count=row["access_count"],
                        tags=row["tags"],
                        metadata=row["metadata"]
                    )
                    memories.append(memory)
                
                return memories
        except Exception as e:
            logger.error(f"Failed to retrieve memory entries: {e}")
            return []
    
    def update_memory_access(self, memory_id: str) -> bool:
        """Update memory entry access count and timestamp"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE memory_entries 
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE id = ?
                """, (datetime.now(), memory_id))
                conn.commit()
                return conn.total_changes > 0
        except Exception as e:
            logger.error(f"Failed to update memory access: {e}")
            return False
    
    # Knowledge graph methods
    def save_graph_node(self, node: GraphNode) -> bool:
        """Save a knowledge graph node"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO graph_nodes (
                        id, label, node_type, properties, created_time, updated_time
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    node.id, node.label, node.node_type, node.properties,
                    node.created_time, node.updated_time
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save graph node: {e}")
            return False
    
    def save_graph_edge(self, edge: GraphEdge) -> bool:
        """Save a knowledge graph edge"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO graph_edges (
                        id, from_node, to_node, relationship_type, weight, properties, created_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    edge.id, edge.from_node, edge.to_node, edge.relationship_type,
                    edge.weight, edge.properties, edge.created_time
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save graph edge: {e}")
            return False
    
    def get_knowledge_graph(self, query: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Retrieve knowledge graph data"""
        try:
            nodes_query = "SELECT * FROM graph_nodes"
            edges_query = "SELECT * FROM graph_edges"
            params = []
            
            if query:
                nodes_query += " WHERE label LIKE ? OR node_type LIKE ?"
                params = [f"%{query}%", f"%{query}%"]
            
            with self._get_connection() as conn:
                # Get nodes
                cursor = conn.execute(nodes_query, params)
                node_rows = cursor.fetchall()
                
                nodes = []
                for row in node_rows:
                    node_data = {
                        "id": row["id"],
                        "label": row["label"],
                        "type": row["node_type"],
                        "properties": json.loads(row["properties"]) if row["properties"] else {},
                        "created_time": row["created_time"],
                        "updated_time": row["updated_time"]
                    }
                    nodes.append(node_data)
                
                # Get edges
                cursor = conn.execute(edges_query)
                edge_rows = cursor.fetchall()
                
                edges = []
                for row in edge_rows:
                    edge_data = {
                        "id": row["id"],
                        "from": row["from_node"],
                        "to": row["to_node"],
                        "label": row["relationship_type"],
                        "weight": row["weight"],
                        "properties": json.loads(row["properties"]) if row["properties"] else {},
                        "created_time": row["created_time"]
                    }
                    edges.append(edge_data)
                
                return {"nodes": nodes, "edges": edges}
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge graph: {e}")
            return {"nodes": [], "edges": []}
    
    # Document tracking methods
    def save_document(self, doc_id: str, filename: str, file_type: str, file_size: int, 
                     processing_options: Dict[str, Any]) -> bool:
        """Save document information"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO documents (
                        id, filename, file_type, file_size, upload_time,
                        processing_status, processing_options
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id, filename, file_type, file_size, datetime.now(),
                    "UPLOADED", json.dumps(processing_options)
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save document: {e}")
            return False
    
    def update_document_status(self, doc_id: str, status: str, results: Optional[Dict] = None) -> bool:
        """Update document processing status"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE documents 
                    SET processing_status = ?, results = ?
                    WHERE id = ?
                """, (status, json.dumps(results) if results else None, doc_id))
                conn.commit()
                return conn.total_changes > 0
        except Exception as e:
            logger.error(f"Failed to update document status: {e}")
            return False
    
    def get_documents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve document records"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM documents 
                    ORDER BY upload_time DESC 
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                
                documents = []
                for row in rows:
                    doc_data = {
                        "id": row["id"],
                        "filename": row["filename"],
                        "file_type": row["file_type"],
                        "file_size": row["file_size"],
                        "upload_time": row["upload_time"],
                        "processing_status": row["processing_status"],
                        "processing_options": json.loads(row["processing_options"]) if row["processing_options"] else {},
                        "results": json.loads(row["results"]) if row["results"] else {}
                    }
                    documents.append(doc_data)
                
                return documents
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    # Logging methods
    def log_system_event(self, level: str, component: str, message: str, 
                        user_id: Optional[str] = None, session_id: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> bool:
        """Log system events"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO system_logs (
                        timestamp, level, component, message, user_id, session_id, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(), level, component, message, user_id, session_id,
                    json.dumps(metadata) if metadata else None
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to log system event: {e}")
            return False
    
    def get_system_logs(self, filters: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve system logs with optional filters"""
        try:
            query = "SELECT * FROM system_logs WHERE 1=1"
            params = []
            
            if filters:
                if "level" in filters and filters["level"] != "ALL":
                    query += " AND level = ?"
                    params.append(filters["level"])
                
                if "component" in filters and filters["component"] != "ALL":
                    query += " AND component = ?"
                    params.append(filters["component"])
                
                if "start_time" in filters:
                    query += " AND timestamp >= ?"
                    params.append(filters["start_time"])
                
                if "end_time" in filters:
                    query += " AND timestamp <= ?"
                    params.append(filters["end_time"])
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            with self._get_connection() as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                logs = []
                for row in rows:
                    log_data = {
                        "id": row["id"],
                        "timestamp": row["timestamp"],
                        "level": row["level"],
                        "component": row["component"],
                        "message": row["message"],
                        "user_id": row["user_id"],
                        "session_id": row["session_id"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                    }
                    logs.append(log_data)
                
                return logs
        except Exception as e:
            logger.error(f"Failed to retrieve system logs: {e}")
            return []
    
    # Analytics methods
    def get_analytics_data(self) -> Dict[str, Any]:
        """Get analytics data for dashboard"""
        try:
            with self._get_connection() as conn:
                # Document stats
                cursor = conn.execute("SELECT COUNT(*) as total FROM documents")
                total_docs = cursor.fetchone()["total"]
                
                cursor = conn.execute("SELECT COUNT(*) as processed FROM documents WHERE processing_status = 'COMPLETED'")
                processed_docs = cursor.fetchone()["processed"]
                
                # Violation stats
                cursor = conn.execute("SELECT COUNT(*) as total FROM violations")
                total_violations = cursor.fetchone()["total"]
                
                cursor = conn.execute("SELECT COUNT(*) as pending FROM violations WHERE status = 'PENDING'")
                pending_violations = cursor.fetchone()["pending"]
                
                # Memory stats
                cursor = conn.execute("SELECT COUNT(*) as total FROM memory_entries")
                total_memory = cursor.fetchone()["total"]
                
                cursor = conn.execute("SELECT AVG(confidence) as avg_conf FROM memory_entries")
                avg_memory_conf = cursor.fetchone()["avg_conf"] or 0
                
                # Graph stats
                cursor = conn.execute("SELECT COUNT(*) as nodes FROM graph_nodes")
                total_nodes = cursor.fetchone()["nodes"]
                
                cursor = conn.execute("SELECT COUNT(*) as edges FROM graph_edges")
                total_edges = cursor.fetchone()["edges"]
                
                return {
                    "documents": {
                        "total": total_docs,
                        "processed": processed_docs,
                        "success_rate": (processed_docs / max(1, total_docs)) * 100
                    },
                    "violations": {
                        "total": total_violations,
                        "pending": pending_violations
                    },
                    "memory": {
                        "total": total_memory,
                        "avg_confidence": avg_memory_conf
                    },
                    "knowledge_graph": {
                        "nodes": total_nodes,
                        "edges": total_edges,
                        "density": total_edges / max(1, total_nodes**2) if total_nodes > 0 else 0
                    }
                }
        except Exception as e:
            logger.error(f"Failed to retrieve analytics data: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 90) -> bool:
        """Clean up old data based on retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self._get_connection() as conn:
                # Clean old logs
                conn.execute("DELETE FROM system_logs WHERE timestamp < ?", (cutoff_date,))
                
                # Clean old completed documents
                conn.execute("""
                    DELETE FROM documents 
                    WHERE upload_time < ? AND processing_status = 'COMPLETED'
                """, (cutoff_date,))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return False