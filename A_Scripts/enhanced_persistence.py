"""
Enhanced Persistence Layer - Addresses Architecture Debt
Provides ACID transactions, connection pooling, and proper database integration
"""

import asyncio
import asyncpg
import aioredis
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import asynccontextmanager
import json
import uuid
from enum import Enum
import structlog
from pathlib import Path

logger = structlog.get_logger()

class EntityStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

@dataclass
class EntityRecord:
    """Database entity record with full audit trail."""
    entity_id: str
    entity_type: str
    canonical_name: str
    attributes: Dict[str, Any]
    confidence_score: float
    status: EntityStatus
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    version: int = 1
    source_documents: List[str] = None
    
    def __post_init__(self):
        if self.source_documents is None:
            self.source_documents = []

@dataclass
class WorkflowRecord:
    """Database workflow record with state tracking."""
    workflow_id: str
    state: str
    documents: Dict[str, Any]
    agent_contexts: Dict[str, Any]
    shared_data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    created_by: str
    total_processing_time: float = 0.0
    completion_percentage: float = 0.0

class ConnectionPool:
    """Manages database connection pooling for high performance."""
    
    def __init__(self, database_url: str, redis_url: str, min_connections: int = 5, max_connections: int = 20):
        self.database_url = database_url
        self.redis_url = redis_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.redis_pool: Optional[aioredis.ConnectionPool] = None
    
    async def initialize(self):
        """Initialize connection pools."""
        try:
            # PostgreSQL connection pool
            self.pg_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_connections,
                max_size=self.max_connections,
                command_timeout=60
            )
            
            # Redis connection pool
            self.redis_pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections
            )
            
            logger.info("database_pools_initialized",
                       pg_min=self.min_connections,
                       pg_max=self.max_connections,
                       redis_max=self.max_connections)
            
        except Exception as e:
            logger.error("pool_initialization_failed", error=str(e))
            raise
    
    @asynccontextmanager
    async def get_pg_connection(self):
        """Get PostgreSQL connection from pool."""
        if not self.pg_pool:
            raise RuntimeError("PostgreSQL pool not initialized")
        
        async with self.pg_pool.acquire() as connection:
            yield connection
    
    @asynccontextmanager
    async def get_redis_connection(self):
        """Get Redis connection from pool."""
        if not self.redis_pool:
            raise RuntimeError("Redis pool not initialized")
        
        redis = aioredis.Redis(connection_pool=self.redis_pool)
        try:
            yield redis
        finally:
            await redis.close()
    
    async def close(self):
        """Close all connection pools."""
        if self.pg_pool:
            await self.pg_pool.close()
        
        if self.redis_pool:
            await self.redis_pool.disconnect()

class TransactionManager:
    """Manages ACID transactions across multiple operations."""
    
    def __init__(self, connection_pool: ConnectionPool):
        self.pool = connection_pool
    
    @asynccontextmanager
    async def transaction(self):
        """Create an ACID transaction context."""
        async with self.pool.get_pg_connection() as connection:
            async with connection.transaction():
                yield connection

class EntityRepository:
    """Repository for entity CRUD operations with full audit trail."""
    
    def __init__(self, connection_pool: ConnectionPool):
        self.pool = connection_pool
        self.transaction_manager = TransactionManager(connection_pool)
    
    async def create_entity(self, entity: EntityRecord) -> str:
        """Create new entity with audit trail."""
        async with self.transaction_manager.transaction() as conn:
            # Insert main entity record
            entity_id = await conn.fetchval("""
                INSERT INTO entities (
                    entity_id, entity_type, canonical_name, attributes, 
                    confidence_score, status, created_at, updated_at, 
                    created_by, updated_by, version
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING entity_id
            """, 
            entity.entity_id, entity.entity_type, entity.canonical_name,
            json.dumps(entity.attributes), entity.confidence_score,
            entity.status.value, entity.created_at, entity.updated_at,
            entity.created_by, entity.updated_by, entity.version)
            
            # Insert source documents
            for doc_id in entity.source_documents:
                await conn.execute("""
                    INSERT INTO entity_documents (entity_id, document_id, created_at)
                    VALUES ($1, $2, $3)
                """, entity_id, doc_id, datetime.now())
            
            logger.info("entity_created",
                       entity_id=entity_id,
                       entity_type=entity.entity_type,
                       created_by=entity.created_by)
            
            return entity_id
    
    async def update_entity(self, entity_id: str, updates: Dict[str, Any], updated_by: str) -> bool:
        """Update entity with version control and audit trail."""
        async with self.transaction_manager.transaction() as conn:
            # Get current version
            current = await conn.fetchrow(
                "SELECT version, updated_at FROM entities WHERE entity_id = $1 AND status != 'deleted'",
                entity_id
            )
            
            if not current:
                logger.warning("entity_not_found_for_update", entity_id=entity_id)
                return False
            
            new_version = current['version'] + 1
            now = datetime.now()
            
            # Build update query dynamically
            set_clauses = []
            values = []
            param_count = 1
            
            for field, value in updates.items():
                if field in ['entity_type', 'canonical_name', 'attributes', 'confidence_score', 'status']:
                    set_clauses.append(f"{field} = ${param_count}")
                    if field == 'attributes':
                        values.append(json.dumps(value))
                    elif field == 'status':
                        values.append(value.value if isinstance(value, EntityStatus) else value)
                    else:
                        values.append(value)
                    param_count += 1
            
            if not set_clauses:
                return False
            
            # Add standard update fields
            set_clauses.extend([
                f"updated_at = ${param_count}",
                f"updated_by = ${param_count + 1}",
                f"version = ${param_count + 2}"
            ])
            values.extend([now, updated_by, new_version])
            
            query = f"""
                UPDATE entities 
                SET {', '.join(set_clauses)}
                WHERE entity_id = ${param_count + 3} AND version = ${param_count + 4}
            """
            values.extend([entity_id, current['version']])
            
            result = await conn.execute(query, *values)
            
            if result == "UPDATE 0":
                logger.warning("entity_update_conflict", 
                             entity_id=entity_id,
                             expected_version=current['version'])
                return False
            
            # Log audit trail
            await conn.execute("""
                INSERT INTO entity_audit_log (
                    entity_id, action, changes, performed_by, performed_at, version
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """, entity_id, "update", json.dumps(updates), updated_by, now, new_version)
            
            logger.info("entity_updated",
                       entity_id=entity_id,
                       version=new_version,
                       updated_by=updated_by)
            
            return True
    
    async def get_entity(self, entity_id: str) -> Optional[EntityRecord]:
        """Get entity by ID with full details."""
        async with self.pool.get_pg_connection() as conn:
            row = await conn.fetchrow("""
                SELECT e.*, COALESCE(
                    array_agg(ed.document_id) FILTER (WHERE ed.document_id IS NOT NULL), 
                    ARRAY[]::text[]
                ) as source_documents
                FROM entities e
                LEFT JOIN entity_documents ed ON e.entity_id = ed.entity_id
                WHERE e.entity_id = $1 AND e.status != 'deleted'
                GROUP BY e.entity_id
            """, entity_id)
            
            if not row:
                return None
            
            return EntityRecord(
                entity_id=row['entity_id'],
                entity_type=row['entity_type'],
                canonical_name=row['canonical_name'],
                attributes=json.loads(row['attributes']),
                confidence_score=row['confidence_score'],
                status=EntityStatus(row['status']),
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                created_by=row['created_by'],
                updated_by=row['updated_by'],
                version=row['version'],
                source_documents=list(row['source_documents'])
            )
    
    async def find_similar_entities(self, entity_type: str, name: str, similarity_threshold: float = 0.8) -> List[EntityRecord]:
        """Find similar entities using database similarity functions."""
        async with self.pool.get_pg_connection() as conn:
            rows = await conn.fetch("""
                SELECT e.*, COALESCE(
                    array_agg(ed.document_id) FILTER (WHERE ed.document_id IS NOT NULL), 
                    ARRAY[]::text[]
                ) as source_documents,
                similarity(e.canonical_name, $2) as sim_score
                FROM entities e
                LEFT JOIN entity_documents ed ON e.entity_id = ed.entity_id
                WHERE e.entity_type = $1 
                AND e.status = 'active'
                AND similarity(e.canonical_name, $2) > $3
                GROUP BY e.entity_id
                ORDER BY sim_score DESC
                LIMIT 10
            """, entity_type, name, similarity_threshold)
            
            entities = []
            for row in rows:
                entities.append(EntityRecord(
                    entity_id=row['entity_id'],
                    entity_type=row['entity_type'],
                    canonical_name=row['canonical_name'],
                    attributes=json.loads(row['attributes']),
                    confidence_score=row['confidence_score'],
                    status=EntityStatus(row['status']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    created_by=row['created_by'],
                    updated_by=row['updated_by'],
                    version=row['version'],
                    source_documents=list(row['source_documents'])
                ))
            
            return entities
    
    async def batch_create_entities(self, entities: List[EntityRecord]) -> List[str]:
        """Batch create entities for performance."""
        if not entities:
            return []
        
        async with self.transaction_manager.transaction() as conn:
            entity_ids = []
            
            # Prepare batch insert data
            entity_data = []
            document_data = []
            
            for entity in entities:
                entity_data.append((
                    entity.entity_id, entity.entity_type, entity.canonical_name,
                    json.dumps(entity.attributes), entity.confidence_score,
                    entity.status.value, entity.created_at, entity.updated_at,
                    entity.created_by, entity.updated_by, entity.version
                ))
                
                for doc_id in entity.source_documents:
                    document_data.append((entity.entity_id, doc_id, datetime.now()))
            
            # Batch insert entities
            await conn.executemany("""
                INSERT INTO entities (
                    entity_id, entity_type, canonical_name, attributes, 
                    confidence_score, status, created_at, updated_at, 
                    created_by, updated_by, version
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, entity_data)
            
            # Batch insert document relationships
            if document_data:
                await conn.executemany("""
                    INSERT INTO entity_documents (entity_id, document_id, created_at)
                    VALUES ($1, $2, $3)
                """, document_data)
            
            entity_ids = [entity.entity_id for entity in entities]
            
            logger.info("entities_batch_created",
                       count=len(entities),
                       entity_ids=entity_ids[:5])  # Log first 5 IDs
            
            return entity_ids

class WorkflowRepository:
    """Repository for workflow state persistence with ACID guarantees."""
    
    def __init__(self, connection_pool: ConnectionPool):
        self.pool = connection_pool
        self.transaction_manager = TransactionManager(connection_pool)
    
    async def save_workflow(self, workflow: WorkflowRecord) -> bool:
        """Save workflow state with atomic operations."""
        async with self.transaction_manager.transaction() as conn:
            # Upsert workflow record
            await conn.execute("""
                INSERT INTO workflows (
                    workflow_id, state, documents, agent_contexts, shared_data,
                    created_at, updated_at, created_by, total_processing_time, completion_percentage
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (workflow_id) DO UPDATE SET
                    state = EXCLUDED.state,
                    documents = EXCLUDED.documents,
                    agent_contexts = EXCLUDED.agent_contexts,
                    shared_data = EXCLUDED.shared_data,
                    updated_at = EXCLUDED.updated_at,
                    total_processing_time = EXCLUDED.total_processing_time,
                    completion_percentage = EXCLUDED.completion_percentage
            """,
            workflow.workflow_id, workflow.state,
            json.dumps(workflow.documents), json.dumps(workflow.agent_contexts),
            json.dumps(workflow.shared_data), workflow.created_at, workflow.updated_at,
            workflow.created_by, workflow.total_processing_time, workflow.completion_percentage)
            
            logger.debug("workflow_saved", workflow_id=workflow.workflow_id)
            return True
    
    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowRecord]:
        """Get workflow by ID."""
        async with self.pool.get_pg_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM workflows WHERE workflow_id = $1",
                workflow_id
            )
            
            if not row:
                return None
            
            return WorkflowRecord(
                workflow_id=row['workflow_id'],
                state=row['state'],
                documents=json.loads(row['documents']),
                agent_contexts=json.loads(row['agent_contexts']),
                shared_data=json.loads(row['shared_data']),
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                created_by=row['created_by'],
                total_processing_time=row['total_processing_time'],
                completion_percentage=row['completion_percentage']
            )
    
    async def get_active_workflows(self) -> List[WorkflowRecord]:
        """Get all active workflows."""
        async with self.pool.get_pg_connection() as conn:
            rows = await conn.fetch("""
                SELECT * FROM workflows 
                WHERE state NOT IN ('completed', 'error', 'cancelled')
                ORDER BY created_at DESC
            """)
            
            workflows = []
            for row in rows:
                workflows.append(WorkflowRecord(
                    workflow_id=row['workflow_id'],
                    state=row['state'],
                    documents=json.loads(row['documents']),
                    agent_contexts=json.loads(row['agent_contexts']),
                    shared_data=json.loads(row['shared_data']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    created_by=row['created_by'],
                    total_processing_time=row['total_processing_time'],
                    completion_percentage=row['completion_percentage']
                ))
            
            return workflows

class CacheManager:
    """Redis-based caching for high-performance operations."""
    
    def __init__(self, connection_pool: ConnectionPool):
        self.pool = connection_pool
        self.default_ttl = 3600  # 1 hour
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        async with self.pool.get_redis_connection() as redis:
            value = await redis.get(key)
            if value:
                return json.loads(value)
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cached value with TTL."""
        async with self.pool.get_redis_connection() as redis:
            ttl = ttl or self.default_ttl
            await redis.setex(key, ttl, json.dumps(value, default=str))
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete cached value."""
        async with self.pool.get_redis_connection() as redis:
            result = await redis.delete(key)
            return result > 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self.pool.get_redis_connection() as redis:
            info = await redis.info()
            return {
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "hit_rate": info.get("keyspace_hits", 0) / (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
            }

class EnhancedPersistenceManager:
    """Central persistence manager coordinating all data operations."""
    
    def __init__(self, database_url: str, redis_url: str):
        self.connection_pool = ConnectionPool(database_url, redis_url)
        self.entity_repo = EntityRepository(self.connection_pool)
        self.workflow_repo = WorkflowRepository(self.connection_pool)
        self.cache_manager = CacheManager(self.connection_pool)
        self.initialized = False
    
    async def initialize(self):
        """Initialize the persistence layer."""
        await self.connection_pool.initialize()
        await self._create_schema()
        self.initialized = True
        
        logger.info("enhanced_persistence_initialized")
    
    async def _create_schema(self):
        """Create database schema if not exists."""
        async with self.connection_pool.get_pg_connection() as conn:
            # Enable similarity extension for fuzzy matching
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            
            # Entities table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    attributes JSONB NOT NULL DEFAULT '{}',
                    confidence_score REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    created_by TEXT NOT NULL,
                    updated_by TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1
                )
            """)
            
            # Entity documents relationship
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_documents (
                    entity_id TEXT REFERENCES entities(entity_id),
                    document_id TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    PRIMARY KEY (entity_id, document_id)
                )
            """)
            
            # Entity audit log
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_audit_log (
                    id SERIAL PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    changes JSONB NOT NULL,
                    performed_by TEXT NOT NULL,
                    performed_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    version INTEGER NOT NULL
                )
            """)
            
            # Workflows table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    documents JSONB NOT NULL DEFAULT '{}',
                    agent_contexts JSONB NOT NULL DEFAULT '{}',
                    shared_data JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    created_by TEXT NOT NULL,
                    total_processing_time REAL NOT NULL DEFAULT 0.0,
                    completion_percentage REAL NOT NULL DEFAULT 0.0
                )
            """)
            
            # Create indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_type_name 
                ON entities USING gin (entity_type, canonical_name gin_trgm_ops)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_status 
                ON entities (status) WHERE status = 'active'
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflows_state 
                ON workflows (state, updated_at)
            """)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all persistence components."""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        try:
            # Test PostgreSQL
            async with self.connection_pool.get_pg_connection() as conn:
                await conn.fetchval("SELECT 1")
            pg_status = "healthy"
        except Exception as e:
            pg_status = f"unhealthy: {str(e)}"
        
        try:
            # Test Redis
            cache_stats = await self.cache_manager.get_cache_stats()
            redis_status = "healthy"
        except Exception as e:
            redis_status = f"unhealthy: {str(e)}"
            cache_stats = {}
        
        return {
            "status": "healthy" if pg_status == "healthy" and redis_status == "healthy" else "degraded",
            "postgresql": pg_status,
            "redis": redis_status,
            "cache_stats": cache_stats,
            "connection_pools": {
                "pg_pool_size": self.connection_pool.pg_pool._holders.__len__() if self.connection_pool.pg_pool else 0,
                "redis_pool_size": len(self.connection_pool.redis_pool._available_connections) if self.connection_pool.redis_pool else 0
            }
        }
    
    async def close(self):
        """Close all connections and clean up."""
        await self.connection_pool.close()
        logger.info("enhanced_persistence_closed")