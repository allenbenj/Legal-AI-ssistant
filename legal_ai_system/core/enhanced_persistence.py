# legal_ai_system/persistence/enhanced_persistence.py

#Enhanced Persistence Layer - Addresses Architecture Debt
#Provides ACID transactions, connection pooling, and proper database integration
#for PostgreSQL and Redis.


import asyncio
import asyncpg # For PostgreSQL
import aioredis # For Redis
import aiofiles
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, asdict, field # Added field
from datetime import datetime, timezone # Added timezone
from contextlib import asynccontextmanager
import json
import uuid
from enum import Enum
# import structlog # Replaced by detailed_logging
from pathlib import Path # Not directly used but good for potential path ops

# Use detailed_logging
from ..core.detailed_logging import get_detailed_logger, LogCategory, detailed_log_function
from ..core.agent_unified_config import _get_service_sync
# Import exceptions
from ..core.unified_exceptions import DatabaseError, ConfigurationError

# Initialize logger for this module
persistence_logger = get_detailed_logger("EnhancedPersistence", LogCategory.DATABASE)

class EntityStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted" # Soft delete

@dataclass
class EntityRecord:
    """Database entity record with full audit trail."""
    entity_id: str # Should be unique, consider UUID
    entity_type: str
    canonical_name: str
    created_by: str # User or system component ID
    updated_by: str # User or system component ID
    attributes: Dict[str, Any] = field(default_factory=dict) # Ensure default factory
    confidence_score: float = 1.0 # Default to 1.0 if not specified
    status: EntityStatus = EntityStatus.ACTIVE # Default status
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    source_documents: List[str] = field(default_factory=list) # Ensure default factory
    
    def __post_init__(self): # Not strictly needed if using default_factory
        if self.source_documents is None: # Should be handled by default_factory
            self.source_documents = []

@dataclass
class RelationshipRecord:
    """Database relationship record."""

    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    created_by: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0
    source_document: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class WorkflowRecord:
    """Database workflow record with state tracking."""
    workflow_id: str # Should be unique, consider UUID
    state: str # Consider Enum for states
    created_by: str
    documents: Dict[str, Any] = field(default_factory=dict)
    agent_contexts: Dict[str, Any] = field(default_factory=dict)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_processing_time: float = 0.0
    completion_percentage: float = 0.0

class ConnectionPool:
    """Manages database connection pooling for PostgreSQL and Redis."""
    
    def __init__(self, database_url: Optional[str], redis_url: Optional[str], 
                 min_pg_connections: int = 5, max_pg_connections: int = 20,
                 max_redis_connections: int = 10): # Added max_redis_connections
        if not database_url and not redis_url:
            raise ConfigurationError("At least one of database_url or redis_url must be provided for ConnectionPool.")
            
        self.database_url = database_url
        self.redis_url = redis_url
        self.min_pg_connections = min_pg_connections
        self.max_pg_connections = max_pg_connections
        self.max_redis_connections = max_redis_connections
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.redis_pool: Optional[aioredis.ConnectionPool] = None # Correct type
        self.logger = persistence_logger.getChild("ConnectionPool") # Specific logger

    @detailed_log_function(LogCategory.DATABASE)
    async def initialize(self):
        """Initialize connection pools."""
        self.logger.info("Initializing database connection pools.")
        try:
            if self.database_url:
                self.pg_pool = await asyncpg.create_pool(
                    dsn=self.database_url, # Use dsn parameter
                    min_size=self.min_pg_connections,
                    max_size=self.max_pg_connections,
                    command_timeout=60,
                    # Consider adding connection setup functions if needed (e.g., for custom types)
                )
                self.logger.info("PostgreSQL connection pool initialized.", 
                                 parameters={'min_con': self.min_pg_connections, 'max_con': self.max_pg_connections})
            else:
                self.logger.warning("PostgreSQL database_url not provided. PostgreSQL pool will not be initialized.")

            if self.redis_url:
                self.redis_pool = aioredis.ConnectionPool.from_url(
                    self.redis_url,
                    max_connections=self.max_redis_connections,
                    # Add timeout settings for Redis if applicable
                    # socket_connect_timeout=5, socket_timeout=5 
                )
                # Test Redis connection
                redis_test_client = aioredis.Redis(connection_pool=self.redis_pool)
                await redis_test_client.ping()
                await redis_test_client.close() # Close the test client
                self.logger.info("Redis connection pool initialized and tested.",
                                 parameters={'max_con': self.max_redis_connections})
            else:
                self.logger.warning("Redis redis_url not provided. Redis pool will not be initialized.")
            
        except Exception as e:
            self.logger.error("Connection pool initialization failed.", exception=e)
            # Decide if this is a fatal error for the application
            raise DatabaseError(f"Failed to initialize connection pools: {str(e)}", cause=e)
    
    @asynccontextmanager
    async def get_pg_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get PostgreSQL connection from pool."""
        if not self.pg_pool:
            self.logger.error("Attempted to get PG connection, but pool is not initialized.")
            raise DatabaseError("PostgreSQL pool not initialized. Check configuration and logs.")
        
        # Type hinting for connection is tricky because Pool.acquire() returns a private proxy.
        async with self.pg_pool.acquire() as connection:
            self.logger.trace("PostgreSQL connection acquired from pool.")
            yield connection
            self.logger.trace("PostgreSQL connection released back to pool.")
    
    @asynccontextmanager
    async def get_redis_connection(self) -> AsyncGenerator[aioredis.Redis, None]:
        """Get Redis connection from pool."""
        if not self.redis_pool:
            self.logger.error("Attempted to get Redis connection, but pool is not initialized.")
            raise DatabaseError("Redis pool not initialized. Check configuration and logs.")
        
        redis = aioredis.Redis(connection_pool=self.redis_pool)
        try:
            self.logger.trace("Redis connection acquired from pool.")
            yield redis
        finally:
            await redis.close() # Ensure client is closed to release connection to pool
            self.logger.trace("Redis connection released back to pool.")
    
    @detailed_log_function(LogCategory.DATABASE)
    async def close(self):
        """Close all connection pools."""
        self.logger.info("Closing database connection pools.")
        if self.pg_pool:
            try:
                await self.pg_pool.close()
                self.logger.info("PostgreSQL connection pool closed.")
            except Exception as e:
                self.logger.error("Error closing PostgreSQL pool.", exception=e)
        
        if self.redis_pool:
            try:
                await self.redis_pool.disconnect(inuse_connections=True) # Ensure connections are closed
                self.logger.info("Redis connection pool disconnected.")
            except Exception as e:
                self.logger.error("Error disconnecting Redis pool.", exception=e)

class TransactionManager:
    """Manages ACID transactions for PostgreSQL operations."""
    
    def __init__(self, connection_pool: ConnectionPool):
        self.pool = connection_pool
        self.logger = persistence_logger.getChild("TransactionManager")

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[asyncpg.Connection, None]: # Changed yield type
        """Create an ACID transaction context for PostgreSQL."""
        self.logger.debug("Starting new database transaction.")
        async with self.pool.get_pg_connection() as connection:
            # Start a new transaction block
            pg_transaction = connection.transaction()
            await pg_transaction.start()
            self.logger.trace("PostgreSQL transaction started.", parameters={'transaction_id': id(pg_transaction)})
            try:
                yield connection # Yield the connection, not the transaction object directly
                await pg_transaction.commit()
                self.logger.debug("Database transaction committed.", parameters={'transaction_id': id(pg_transaction)})
            except Exception as e:
                self.logger.error("Database transaction failed, rolling back.", 
                                 parameters={'transaction_id': id(pg_transaction)}, exception=e)
                if not pg_transaction.is_completed(): # Check if rollback is possible
                    await pg_transaction.rollback()
                    self.logger.info("Database transaction rolled back.", parameters={'transaction_id': id(pg_transaction)})
                # Re-raise the original exception after rollback attempt
                if isinstance(e, asyncpg.PostgresError):
                    raise DatabaseError(f"PostgreSQL transaction error: {str(e)}", database_type="postgresql", cause=e)
                raise # Re-raise other exceptions

class EntityRepository:
    """Repository for entity CRUD operations with full audit trail."""
    
    def __init__(self, connection_pool: ConnectionPool):
        self.pool = connection_pool
        self.transaction_manager = TransactionManager(connection_pool)
        self.logger = persistence_logger.getChild("EntityRepository")
    
    @detailed_log_function(LogCategory.DATABASE)
    async def create_entity(self, entity: EntityRecord) -> str:
        """Create new entity with audit trail."""
        self.logger.info("Creating new entity.", parameters={'entity_type': entity.entity_type, 'name': entity.canonical_name})
        try:
            async with self.transaction_manager.transaction() as conn:
                # Ensure entity_id is set, generate if not
                if not entity.entity_id:
                    entity.entity_id = str(uuid.uuid4())
                    self.logger.debug("Generated UUID for new entity.", parameters={'entity_id': entity.entity_id})

                # Insert main entity record
                # Using named placeholders for asyncpg ($1, $2, etc.)
                entity_id_returned = await conn.fetchval("""
                    INSERT INTO entities (
                        entity_id, entity_type, canonical_name, attributes, 
                        confidence_score, status, created_at, updated_at, 
                        created_by, updated_by, version
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING entity_id
                """, 
                entity.entity_id, entity.entity_type, entity.canonical_name,
                json.dumps(entity.attributes), entity.confidence_score, # Attributes as JSONB
                entity.status.value, entity.created_at, entity.updated_at,
                entity.created_by, entity.updated_by, entity.version)
                
                if not entity_id_returned: # Should not happen if INSERT is successful and RETURNING is used
                    raise DatabaseError("Entity creation did not return an ID.")
                
                # Insert source documents if any
                if entity.source_documents:
                    doc_data_to_insert = [(entity.entity_id, doc_id, datetime.now(timezone.utc)) for doc_id in entity.source_documents]
                    if doc_data_to_insert:
                        await conn.executemany("""
                            INSERT INTO entity_documents (entity_id, document_id, created_at)
                            VALUES ($1, $2, $3) ON CONFLICT (entity_id, document_id) DO NOTHING
                        """, doc_data_to_insert)
                
                self.logger.info("Entity created successfully in DB.",
                                parameters={'entity_id': entity.entity_id, 'entity_type': entity.entity_type})
                return entity.entity_id # Return the original or generated ID
        except asyncpg.PostgresError as e:
            self.logger.error("Database error during entity creation.", exception=e)
            raise DatabaseError(f"Failed to create entity '{entity.canonical_name}': {str(e)}", database_type="postgresql", cause=e)
        except Exception as e: # Catch other unexpected errors
            self.logger.error("Unexpected error during entity creation.", exception=e)
            raise DatabaseError(f"Unexpected error creating entity '{entity.canonical_name}'", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def update_entity(self, entity_id: str, updates: Dict[str, Any], updated_by: str) -> bool:
        """Update entity with version control and audit trail."""
        self.logger.info("Updating entity.", parameters={'entity_id': entity_id, 'num_updates': len(updates)})
        if not updates:
            self.logger.warning("No updates provided for entity.", parameters={'entity_id': entity_id})
            return False

        try:
            async with self.transaction_manager.transaction() as conn:
                current = await conn.fetchrow(
                    "SELECT version, attributes FROM entities WHERE entity_id = $1 AND status != $2 FOR UPDATE", # Lock row
                    entity_id, EntityStatus.DELETED.value
                )
                
                if not current:
                    self.logger.warning("Entity not found or deleted, cannot update.", parameters={'entity_id': entity_id})
                    return False
                
                current_version = current['version']
                current_attributes = json.loads(current['attributes']) if current['attributes'] else {}
                new_version = current_version + 1
                now_utc = datetime.now(timezone.utc)
                
                set_clauses = []
                values = []
                
                # Merge attributes if 'attributes' is in updates
                if 'attributes' in updates and isinstance(updates['attributes'], dict):
                    current_attributes.update(updates['attributes'])
                    updates['attributes'] = current_attributes # Replace with merged attributes

                for i, (field, value) in enumerate(updates.items()):
                    # Ensure field is valid and prevent SQL injection by not using f-strings for field names directly in SQL
                    # This part needs a mapping of allowed fields to prevent arbitrary field updates.
                    allowed_fields = ['entity_type', 'canonical_name', 'attributes', 'confidence_score', 'status', 'source_documents']
                    if field not in allowed_fields:
                        self.logger.warning(f"Attempt to update disallowed field '{field}' for entity {entity_id}. Skipping.")
                        continue

                    set_clauses.append(f"{field} = ${i+1}")
                    if field == 'attributes': values.append(json.dumps(value))
                    elif field == 'status': values.append(value.value if isinstance(value, EntityStatus) else str(value))
                    elif field == 'source_documents' and isinstance(value, list):
                        # Handle source_documents separately if it's a separate table
                        # For now, assuming it's a JSONB array in 'entities' table for simplicity of this example
                        # This part needs to align with your actual schema for 'source_documents'
                        # If it's a JSONB array in 'entities':
                        # values.append(json.dumps(value))
                        # If it's a separate table (entity_documents), this needs different handling:
                        # 1. Delete existing entity_documents for this entity_id
                        # 2. Insert new ones.
                        # This example will skip direct update of source_documents here and assume it's handled by a dedicated method.
                        self.logger.debug(f"Field '{field}' update skipped in generic method; use dedicated method if needed.",
                                         parameters={'field': field})
                        set_clauses.pop() # Remove the clause
                        continue # Skip adding to values for this field
                    else: values.append(value)
                
                if not set_clauses: # No valid fields to update
                    self.logger.warning("No valid fields to update for entity.", parameters={'entity_id': entity_id})
                    return False

                # Add standard update fields to SET clauses and values list
                param_start_idx = len(values) + 1
                set_clauses.extend([
                    f"updated_at = ${param_start_idx}",
                    f"updated_by = ${param_start_idx + 1}",
                    f"version = ${param_start_idx + 2}"
                ])
                values.extend([now_utc, updated_by, new_version])
                
                # Add WHERE clause parameters
                where_param_start_idx = param_start_idx + 3
                query = f"""
                    UPDATE entities 
                    SET {', '.join(set_clauses)}
                    WHERE entity_id = ${where_param_start_idx} AND version = ${where_param_start_idx + 1}
                """
                values.extend([entity_id, current_version])
                
                result = await conn.execute(query, *values)
                
                # Check if update was successful (optimistic locking check)
                if result == "UPDATE 0":
                    self.logger.warning("Entity update conflict (version mismatch or entity gone).", 
                                     parameters={'entity_id': entity_id, 'expected_version': current_version})
                    raise DatabaseError(f"Update conflict for entity {entity_id}, version {current_version}. Please refresh and retry.")
                
                # Log audit trail
                await conn.execute("""
                    INSERT INTO entity_audit_log (
                        entity_id, action, changes, performed_by, performed_at, version
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, entity_id, "update", json.dumps(updates), updated_by, now_utc, new_version)
                
                self.logger.info("Entity updated successfully in DB.",
                                parameters={'entity_id': entity_id, 'new_version': new_version})
                return True
        except asyncpg.PostgresError as e:
            self.logger.error("Database error during entity update.", parameters={'entity_id': entity_id}, exception=e)
            raise DatabaseError(f"Failed to update entity '{entity_id}': {str(e)}", database_type="postgresql", cause=e)
        except Exception as e:
            self.logger.error("Unexpected error during entity update.", parameters={'entity_id': entity_id}, exception=e)
            raise DatabaseError(f"Unexpected error updating entity '{entity_id}'", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def get_entity(self, entity_id: str) -> Optional[EntityRecord]:
        """Get entity by ID with full details including source documents."""
        self.logger.debug("Fetching entity.", parameters={'entity_id': entity_id})
        try:
            async with self.pool.get_pg_connection() as conn:
                row = await conn.fetchrow("""
                    SELECT e.*, COALESCE(
                        (SELECT array_agg(ed.document_id) FROM entity_documents ed WHERE ed.entity_id = e.entity_id), 
                        ARRAY[]::text[]
                    ) as source_documents_agg
                    FROM entities e
                    WHERE e.entity_id = $1 AND e.status != $2
                """, entity_id, EntityStatus.DELETED.value)
                
                if not row:
                    self.logger.debug("Entity not found.", parameters={'entity_id': entity_id})
                    return None
                
                entity = EntityRecord(
                    entity_id=row['entity_id'],
                    entity_type=row['entity_type'],
                    canonical_name=row['canonical_name'],
                    attributes=json.loads(row['attributes']) if isinstance(row['attributes'], str) else row['attributes'], # Handle JSONB directly
                    confidence_score=row['confidence_score'],
                    status=EntityStatus(row['status']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    created_by=row['created_by'],
                    updated_by=row['updated_by'],
                    version=row['version'],
                    source_documents=list(row['source_documents_agg'])
                )
                self.logger.info("Entity fetched successfully.", parameters={'entity_id': entity_id, 'type': entity.entity_type})
                return entity
        except asyncpg.PostgresError as e:
            self.logger.error("Database error fetching entity.", parameters={'entity_id': entity_id}, exception=e)
            raise DatabaseError(f"Failed to fetch entity '{entity_id}': {str(e)}", database_type="postgresql", cause=e)
        except Exception as e:
            self.logger.error("Unexpected error fetching entity.", parameters={'entity_id': entity_id}, exception=e)
            raise DatabaseError(f"Unexpected error fetching entity '{entity_id}'", cause=e)


    @detailed_log_function(LogCategory.DATABASE)
    async def find_similar_entities(self, entity_type: str, name: str, 
                                  similarity_threshold: float = 0.7, limit: int = 10) -> List[EntityRecord]: # Adjusted threshold
        """Find similar entities using PostgreSQL's pg_trgm similarity."""
        self.logger.debug("Finding similar entities.", parameters={'type': entity_type, 'name_query': name, 'threshold': similarity_threshold})
        try:
            async with self.pool.get_pg_connection() as conn:
                # Ensure pg_trgm extension is enabled (usually done in _create_schema)
                # await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;") # Idempotent

                rows = await conn.fetch("""
                    SELECT e.*, COALESCE(
                        (SELECT array_agg(ed.document_id) FROM entity_documents ed WHERE ed.entity_id = e.entity_id), 
                        ARRAY[]::text[]
                    ) as source_documents_agg,
                    similarity(e.canonical_name, $2) as similarity_score_val
                    FROM entities e
                    WHERE e.entity_type = $1 
                      AND e.status = $4 -- Only search active entities
                      AND similarity(e.canonical_name, $2) >= $3 -- Use pg_trgm similarity
                    GROUP BY e.entity_id -- Ensure GROUP BY includes all non-aggregated selected columns from 'e'
                    ORDER BY similarity_score_val DESC
                    LIMIT $5
                """, entity_type, name, similarity_threshold, EntityStatus.ACTIVE.value, limit)
                
                entities = []
                for row in rows:
                    entities.append(EntityRecord(
                        entity_id=row['entity_id'],
                        entity_type=row['entity_type'],
                        canonical_name=row['canonical_name'],
                        attributes=json.loads(row['attributes']) if isinstance(row['attributes'], str) else row['attributes'],
                        confidence_score=row['confidence_score'],
                        status=EntityStatus(row['status']),
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        created_by=row['created_by'],
                        updated_by=row['updated_by'],
                        version=row['version'],
                        source_documents=list(row['source_documents_agg'])
                    ))
                self.logger.info(f"Found {len(entities)} similar entities.")
                return entities
        except asyncpg.PostgresError as e:
            self.logger.error("Database error finding similar entities.", exception=e)
            raise DatabaseError(f"Failed to find similar entities: {str(e)}", database_type="postgresql", cause=e)
        except Exception as e:
            self.logger.error("Unexpected error finding similar entities.", exception=e)
            raise DatabaseError(f"Unexpected error finding similar entities", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def batch_create_entities(self, entities: List[EntityRecord]) -> List[str]:
        """Batch create entities for performance, returning list of created/existing entity IDs."""
        if not entities: return []
        self.logger.info(f"Batch creating {len(entities)} entities.")
        
        created_entity_ids: List[str] = []
        try:
            async with self.transaction_manager.transaction() as conn:
                entity_insert_data = []
                doc_insert_data = []
                
                for entity in entities:
                    if not entity.entity_id: entity.entity_id = str(uuid.uuid4())
                    entity_insert_data.append((
                        entity.entity_id, entity.entity_type, entity.canonical_name,
                        json.dumps(entity.attributes), entity.confidence_score,
                        entity.status.value, entity.created_at, entity.updated_at,
                        entity.created_by, entity.updated_by, entity.version
                    ))
                    for doc_id in entity.source_documents:
                        doc_insert_data.append((entity.entity_id, doc_id, datetime.now(timezone.utc)))
                    created_entity_ids.append(entity.entity_id)

                # Batch insert entities with ON CONFLICT DO NOTHING to handle potential duplicates if IDs are pre-generated
                # Or use ON CONFLICT DO UPDATE if merging is desired for pre-existing IDs.
                # For now, assuming IDs are new or we want to skip if ID exists.
                await conn.copy_records_to_table(
                    'entities',
                    records=entity_insert_data,
                    columns=[
                        'entity_id', 'entity_type', 'canonical_name', 'attributes', 
                        'confidence_score', 'status', 'created_at', 'updated_at', 
                        'created_by', 'updated_by', 'version'
                    ],
                    # For ON CONFLICT with copy_records_to_table, you might need a temporary table approach
                    # or handle conflicts by checking existence before adding to entity_insert_data.
                    # A simpler robust approach for batch with conflict handling is executemany with ON CONFLICT.
                    # Since copy_records_to_table is faster but less flexible with conflicts, let's use executemany here.
                )
                # Using executemany for better conflict handling:
                # Note: executemany is generally slower than COPY for very large batches.
                # For optimal performance with conflicts, a temp table + merge strategy is best.
                # Here's a simplified executemany with ON CONFLICT DO UPDATE:
                await conn.executemany("""
                    INSERT INTO entities (
                        entity_id, entity_type, canonical_name, attributes, 
                        confidence_score, status, created_at, updated_at, 
                        created_by, updated_by, version
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (entity_id) DO UPDATE SET
                        canonical_name = EXCLUDED.canonical_name,
                        attributes = EXCLUDED.attributes,
                        confidence_score = GREATEST(entities.confidence_score, EXCLUDED.confidence_score),
                        status = EXCLUDED.status,
                        updated_at = EXCLUDED.updated_at,
                        updated_by = EXCLUDED.updated_by,
                        version = entities.version + 1 
                        -- Only update if new data is "better" or different (e.g. newer updated_at)
                        WHERE entities.updated_at < EXCLUDED.updated_at; 
                """, entity_insert_data)


                if doc_insert_data:
                    await conn.executemany("""
                        INSERT INTO entity_documents (entity_id, document_id, created_at)
                        VALUES ($1, $2, $3) ON CONFLICT (entity_id, document_id) DO NOTHING
                    """, doc_insert_data)
                
                self.logger.info("Entities batch processed.", parameters={'count': len(entities)})
                return created_entity_ids # IDs of entities attempted to be created/updated
        except asyncpg.PostgresError as e:
            self.logger.error("Database error during batch entity creation.", exception=e)
            raise DatabaseError(f"Failed to batch create entities: {str(e)}", database_type="postgresql", cause=e)
        except Exception as e:
            self.logger.error("Unexpected error during batch entity creation.", exception=e)
            raise DatabaseError(f"Unexpected error batch creating entities", cause=e)


class RelationshipRepository:
    """Repository for relationship persistence."""

    def __init__(self, connection_pool: ConnectionPool):
        self.pool = connection_pool
        self.transaction_manager = TransactionManager(connection_pool)
        self.logger = persistence_logger.getChild("RelationshipRepository")

    @detailed_log_function(LogCategory.DATABASE)
    async def create_relationship(self, relationship: RelationshipRecord) -> str:
        """Create a relationship entry."""
        self.logger.info(
            "Creating relationship.",
            parameters={
                "src": relationship.source_entity_id,
                "tgt": relationship.target_entity_id,
                "type": relationship.relationship_type,
            },
        )
        try:
            async with self.transaction_manager.transaction() as conn:
                if not relationship.relationship_id:
                    relationship.relationship_id = str(uuid.uuid4())

                await conn.execute(
                    """
                    INSERT INTO relationships (
                        relationship_id, source_entity_id, target_entity_id,
                        relationship_type, attributes, confidence_score,
                        source_document, created_at, created_by
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,
                    relationship.relationship_id,
                    relationship.source_entity_id,
                    relationship.target_entity_id,
                    relationship.relationship_type,
                    json.dumps(relationship.attributes),
                    relationship.confidence_score,
                    relationship.source_document,
                    relationship.created_at,
                    relationship.created_by,
                )

                self.logger.info(
                    "Relationship created.",
                    parameters={"relationship_id": relationship.relationship_id},
                )
                return relationship.relationship_id
        except asyncpg.PostgresError as e:
            self.logger.error("Database error creating relationship.", exception=e)
            raise DatabaseError(
                f"Failed to create relationship: {str(e)}",
                database_type="postgresql",
                cause=e,
            )
        except Exception as e:
            self.logger.error("Unexpected error creating relationship.", exception=e)
            raise DatabaseError("Unexpected error creating relationship", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def get_relationships_for_entity(self, entity_id: str) -> List[RelationshipRecord]:
        """Retrieve all relationships for a given entity."""
        self.logger.debug("Fetching relationships for entity.", parameters={"entity_id": entity_id})
        try:
            async with self.pool.get_pg_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM relationships
                    WHERE source_entity_id = $1 OR target_entity_id = $1
                    """,
                    entity_id,
                )

                relationships = [
                    RelationshipRecord(
                        relationship_id=r["relationship_id"],
                        source_entity_id=r["source_entity_id"],
                        target_entity_id=r["target_entity_id"],
                        relationship_type=r["relationship_type"],
                        attributes=json.loads(r["attributes"]) if isinstance(r["attributes"], str) else r["attributes"],
                        confidence_score=r["confidence_score"],
                        source_document=r["source_document"],
                        created_at=r["created_at"],
                        created_by=r["created_by"],
                    )
                    for r in rows
                ]

                return relationships
        except asyncpg.PostgresError as e:
            self.logger.error("Database error fetching relationships.", exception=e)
            raise DatabaseError(
                f"Failed to fetch relationships for entity {entity_id}: {str(e)}",
                database_type="postgresql",
                cause=e,
            )
        except Exception as e:
            self.logger.error("Unexpected error fetching relationships.", exception=e)
            raise DatabaseError("Unexpected error fetching relationships", cause=e)


class WorkflowRepository:
    """Repository for workflow state persistence with ACID guarantees."""
    
    def __init__(self, connection_pool: ConnectionPool):
        self.pool = connection_pool
        self.transaction_manager = TransactionManager(connection_pool)
        self.logger = persistence_logger.getChild("WorkflowRepository")

    @detailed_log_function(LogCategory.DATABASE)
    async def save_workflow(self, workflow: WorkflowRecord) -> bool:
        """Save or update workflow state with atomic operations."""
        self.logger.info("Saving workflow state.", parameters={'workflow_id': workflow.workflow_id, 'state': workflow.state})
        try:
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
                json.dumps(workflow.documents), json.dumps(workflow.agent_contexts), # JSONB fields
                json.dumps(workflow.shared_data), workflow.created_at, workflow.updated_at,
                workflow.created_by, workflow.total_processing_time, workflow.completion_percentage)
                
                self.logger.info("Workflow state saved successfully.", parameters={'workflow_id': workflow.workflow_id})
                return True
        except asyncpg.PostgresError as e:
            self.logger.error("Database error saving workflow state.", parameters={'workflow_id': workflow.workflow_id}, exception=e)
            raise DatabaseError(f"Failed to save workflow '{workflow.workflow_id}': {str(e)}", database_type="postgresql", cause=e)
        except Exception as e:
            self.logger.error("Unexpected error saving workflow state.", parameters={'workflow_id': workflow.workflow_id}, exception=e)
            raise DatabaseError(f"Unexpected error saving workflow '{workflow.workflow_id}'", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowRecord]:
        """Get workflow by ID."""
        self.logger.debug("Fetching workflow.", parameters={'workflow_id': workflow_id})
        try:
            async with self.pool.get_pg_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM workflows WHERE workflow_id = $1",
                    workflow_id
                )
                
                if not row:
                    self.logger.debug("Workflow not found.", parameters={'workflow_id': workflow_id})
                    return None
                
                workflow = WorkflowRecord(
                    workflow_id=row['workflow_id'], state=row['state'],
                    documents=json.loads(row['documents']) if isinstance(row['documents'], str) else row['documents'],
                    agent_contexts=json.loads(row['agent_contexts']) if isinstance(row['agent_contexts'], str) else row['agent_contexts'],
                    shared_data=json.loads(row['shared_data']) if isinstance(row['shared_data'], str) else row['shared_data'],
                    created_at=row['created_at'], updated_at=row['updated_at'], created_by=row['created_by'],
                    total_processing_time=row['total_processing_time'], completion_percentage=row['completion_percentage']
                )
                self.logger.info("Workflow fetched successfully.", parameters={'workflow_id': workflow_id})
                return workflow
        except asyncpg.PostgresError as e:
            self.logger.error("Database error fetching workflow.", parameters={'workflow_id': workflow_id}, exception=e)
            raise DatabaseError(f"Failed to fetch workflow '{workflow_id}': {str(e)}", database_type="postgresql", cause=e)
        except Exception as e:
            self.logger.error("Unexpected error fetching workflow.", parameters={'workflow_id': workflow_id}, exception=e)
            raise DatabaseError(f"Unexpected error fetching workflow '{workflow_id}'", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def get_active_workflows(self, limit: int = 100) -> List[WorkflowRecord]: # Added limit
        """Get all active (not completed, errored, or cancelled) workflows."""
        self.logger.debug("Fetching active workflows.", parameters={'limit': limit})
        try:
            async with self.pool.get_pg_connection() as conn:
                # Define what states are considered "non-active"
                non_active_states = ['completed', 'error', 'cancelled'] # Case-insensitive if DB is
                
                # Using a placeholder for each item in the list for the query
                placeholders = ', '.join([f'${i+1}' for i in range(len(non_active_states))])
                
                query = f"""
                    SELECT * FROM workflows 
                    WHERE state NOT IN ({placeholders})
                    ORDER BY created_at DESC
                    LIMIT ${len(non_active_states) + 1} 
                """
                params = non_active_states + [limit]
                rows = await conn.fetch(query, *params)
                
                workflows = [
                    WorkflowRecord(
                        workflow_id=row['workflow_id'], state=row['state'],
                        documents=json.loads(row['documents']) if isinstance(row['documents'], str) else row['documents'],
                        agent_contexts=json.loads(row['agent_contexts']) if isinstance(row['agent_contexts'], str) else row['agent_contexts'],
                        shared_data=json.loads(row['shared_data']) if isinstance(row['shared_data'], str) else row['shared_data'],
                        created_at=row['created_at'], updated_at=row['updated_at'], created_by=row['created_by'],
                        total_processing_time=row['total_processing_time'], completion_percentage=row['completion_percentage']
                    ) for row in rows
                ]
                self.logger.info(f"Fetched {len(workflows)} active workflows.")
                return workflows
        except asyncpg.PostgresError as e:
            self.logger.error("Database error fetching active workflows.", exception=e)
            raise DatabaseError(f"Failed to fetch active workflows: {str(e)}", database_type="postgresql", cause=e)
        except Exception as e:
            self.logger.error("Unexpected error fetching active workflows.", exception=e)
            raise DatabaseError(f"Unexpected error fetching active workflows", cause=e)


class CacheManager:
    """Redis-based caching for high-performance operations."""
    
    def __init__(self, connection_pool: ConnectionPool, default_ttl_seconds: int = 3600): # Renamed param
        self.pool = connection_pool
        self.default_ttl = default_ttl_seconds  # 1 hour
        self.logger = persistence_logger.getChild("CacheManager")

    @detailed_log_function(LogCategory.DATABASE)
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value. Returns None if key not found or on error."""
        self.logger.debug("Getting value from cache.", parameters={'key': key})
        if not self.pool.redis_pool: 
            self.logger.warning("Redis pool not initialized, cannot get from cache.")
            return None
        try:
            async with self.pool.get_redis_connection() as redis:
                value_bytes = await redis.get(key) # Returns bytes or None
                if value_bytes:
                    try:
                        value_str = value_bytes.decode('utf-8')
                        deserialized_value = json.loads(value_str)
                        self.logger.info("Cache hit.", parameters={'key': key})
                        return deserialized_value
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        self.logger.error("Failed to decode/deserialize cached value. Invalidating.", 
                                         parameters={'key': key}, exception=e)
                        await redis.delete(key) # Remove corrupted entry
                        return None
                else:
                    self.logger.debug("Cache miss.", parameters={'key': key})
                    return None
        except aioredis.RedisError as e:
            self.logger.error("Redis error during cache get.", parameters={'key': key}, exception=e)
            return None # Treat Redis errors as cache miss
        except Exception as e: # Catch other unexpected errors
            self.logger.error("Unexpected error during cache get.", parameters={'key': key}, exception=e)
            return None


    @detailed_log_function(LogCategory.DATABASE)
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool: # Renamed param
        """Set cached value with TTL. Returns True on success, False on failure."""
        self.logger.debug("Setting value in cache.", parameters={'key': key, 'ttl_sec': ttl_seconds or self.default_ttl})
        if not self.pool.redis_pool: 
            self.logger.warning("Redis pool not initialized, cannot set cache value.")
            return False
        try:
            async with self.pool.get_redis_connection() as redis:
                effective_ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
                # Ensure value is JSON serializable
                try:
                    value_str = json.dumps(value, default=str) # Use default=str for non-serializable like datetime
                except TypeError as te:
                    self.logger.error("Failed to serialize value for cache. Value not set.", 
                                     parameters={'key': key, 'value_type': type(value).__name__}, exception=te)
                    return False
                
                await redis.setex(key, effective_ttl, value_str)
                self.logger.info("Value set in cache successfully.", parameters={'key': key})
                return True
        except aioredis.RedisError as e:
            self.logger.error("Redis error during cache set.", parameters={'key': key}, exception=e)
            return False
        except Exception as e:
            self.logger.error("Unexpected error during cache set.", parameters={'key': key}, exception=e)
            return False


    @detailed_log_function(LogCategory.DATABASE)
    async def delete(self, key: str) -> bool:
        """Delete cached value. Returns True if key was deleted, False otherwise."""
        self.logger.debug("Deleting value from cache.", parameters={'key': key})
        if not self.pool.redis_pool: 
            self.logger.warning("Redis pool not initialized, cannot delete from cache.")
            return False
        try:
            async with self.pool.get_redis_connection() as redis:
                result_int = await redis.delete(key) # Returns number of keys deleted
                was_deleted = result_int > 0
                if was_deleted:
                    self.logger.info("Value deleted from cache.", parameters={'key': key})
                else:
                    self.logger.debug("Value not found in cache for deletion.", parameters={'key': key})
                return was_deleted
        except aioredis.RedisError as e:
            self.logger.error("Redis error during cache delete.", parameters={'key': key}, exception=e)
            return False
        except Exception as e:
            self.logger.error("Unexpected error during cache delete.", parameters={'key': key}, exception=e)
            return False


    @detailed_log_function(LogCategory.DATABASE)
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from Redis INFO command."""
        self.logger.debug("Fetching Redis cache statistics.")
        if not self.pool.redis_pool: 
            self.logger.warning("Redis pool not initialized, cannot get cache stats.")
            return {"error": "Redis pool not initialized"}
        try:
            async with self.pool.get_redis_connection() as redis:
                info = await redis.info() # Returns a dict
                
                # Extract relevant stats
                stats = {
                    "redis_version": info.get("redis_version"),
                    "uptime_in_seconds": info.get("uptime_in_seconds"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human"),
                    "used_memory_peak_human": info.get("used_memory_peak_human"),
                    "total_keys": info.get("db0", {}).get("keys"), # Assuming using default DB 0
                    "keyspace_hits": info.get("keyspace_hits"),
                    "keyspace_misses": info.get("keyspace_misses"),
                }
                hits = info.get("keyspace_hits", 0)
                misses = info.get("keyspace_misses", 0)
                stats["hit_rate"] = (hits / (hits + misses)) if (hits + misses) > 0 else 0.0
                
                self.logger.info("Redis cache statistics retrieved.", parameters={'keys': stats.get('total_keys')})
                return stats
        except aioredis.RedisError as e:
            self.logger.error("Redis error fetching cache stats.", exception=e)
            return {"error": f"Redis error: {str(e)}"}
        except Exception as e:
            self.logger.error("Unexpected error fetching cache stats.", exception=e)
            return {"error": f"Unexpected error: {str(e)}"}

    @detailed_log_function(LogCategory.DATABASE)
    async def persist_to_disk(self, file_path: str) -> bool:
        """Persist all cached values to a JSON file for durability."""
        if not self.pool.redis_pool:
            self.logger.warning("Redis pool not initialized, cannot persist cache.")
            return False
        try:
            async with self.pool.get_redis_connection() as redis:
                keys = await redis.keys("*")
                data: Dict[str, Any] = {}
                for key in keys:
                    val = await redis.get(key)
                    if val is not None:
                        try:
                            data[key.decode("utf-8") if isinstance(key, bytes) else key] = json.loads(val.decode("utf-8"))
                        except Exception:
                            continue
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(data))
            self.logger.info("Cache persisted to disk.", parameters={"file": file_path, "entries": len(data)})
            return True
        except Exception as e:
            self.logger.error("Failed to persist cache to disk.", parameters={"file": file_path}, exception=e)
            return False

    @detailed_log_function(LogCategory.DATABASE)
    async def load_from_disk(self, file_path: str) -> int:
        """Load cached values from a JSON file. Returns number of entries restored."""
        if not self.pool.redis_pool:
            self.logger.warning("Redis pool not initialized, cannot load cache.")
            return 0
        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
            data = json.loads(content)
            async with self.pool.get_redis_connection() as redis:
                for key, value in data.items():
                    await redis.set(key, json.dumps(value))
            self.logger.info("Cache loaded from disk.", parameters={"file": file_path, "entries": len(data)})
            return len(data)
        except FileNotFoundError:
            self.logger.warning("Cache file not found during load.", parameters={"file": file_path})
            return 0
        except Exception as e:
            self.logger.error("Failed to load cache from disk.", parameters={"file": file_path}, exception=e)
            return 0


class EnhancedPersistenceManager:
    """Central persistence manager coordinating all data operations."""

    def __init__(
        self,
        *,
        config: Optional[Dict[str, Any]] = None,
        connection_pool: Optional[ConnectionPool] = None,
        metrics_exporter: Optional[Any] = None,
    ) -> None:
        """Create a new ``EnhancedPersistenceManager`` instance."""

        self.config = config or {}
        cache_ttl = self.config.get("cache_default_ttl_seconds", 3600)

        if connection_pool is None:
            connection_pool = ConnectionPool(
                self.config.get("database_url"),
                self.config.get("redis_url"),
                self.config.get("min_pg_connections", 5),
                self.config.get("max_pg_connections", 20),
                self.config.get("max_redis_connections", 10),
            )
        elif not isinstance(connection_pool, ConnectionPool):
            raise TypeError("connection_pool must be a ConnectionPool instance")

        self.connection_pool = connection_pool
        self.entity_repo = EntityRepository(self.connection_pool)
        self.relationship_repo = RelationshipRepository(self.connection_pool)
        self.workflow_repo = WorkflowRepository(self.connection_pool)
        self.cache_manager = CacheManager(self.connection_pool, cache_ttl)
        self.metrics = metrics_exporter
        self.initialized = False
        self.logger = persistence_logger.getChild("Manager")

    @detailed_log_function(LogCategory.DATABASE)
    async def initialize(self):
        """Initialize the persistence layer, including pools and schema."""
        if self.initialized:
            self.logger.warning("EnhancedPersistenceManager already initialized.")
            return

        self.logger.info("Initializing EnhancedPersistenceManager.")
        try:
            await self.connection_pool.initialize()
            if self.connection_pool.pg_pool: # Only create schema if PG pool is up
                await self._create_schema()
            else:
                self.logger.warning("PostgreSQL pool not initialized. Skipping schema creation. Database operations might fail.")
            
            self.initialized = True
            self.logger.info("EnhancedPersistenceManager initialized successfully.")
        except Exception as e: # Catch broader exceptions from pool/schema init
            self.logger.critical("Failed to initialize EnhancedPersistenceManager.", exception=e)
            # Depending on policy, either raise or allow degraded mode
            # For now, mark as not initialized and raise to prevent use.
            self.initialized = False
            raise DatabaseError("EnhancedPersistenceManager initialization failed.", cause=e)
    
    async def _create_schema(self):
        """Create database schema if not exists."""
        self.logger.info("Creating/Verifying database schema.")
        try:
            async with self.connection_pool.get_pg_connection() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;") # For similarity search
                
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS entities (
                        entity_id TEXT PRIMARY KEY,
                        entity_type TEXT NOT NULL,
                        canonical_name TEXT NOT NULL,
                        attributes JSONB DEFAULT '{}'::jsonb,
                        confidence_score REAL DEFAULT 1.0,
                        status TEXT NOT NULL DEFAULT 'active',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        created_by TEXT NOT NULL,
                        updated_by TEXT NOT NULL,
                        version INTEGER DEFAULT 1
                    );
                """)
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_status ON entities(status);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name_trgm ON entities USING gin (canonical_name gin_trgm_ops);") # For similarity

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS entity_documents (
                        entity_id TEXT REFERENCES entities(entity_id) ON DELETE CASCADE,
                        document_id TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (entity_id, document_id)
                    );
                """)
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_documents_doc_id ON entity_documents(document_id);")

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS entity_audit_log (
                        log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        entity_id TEXT NOT NULL, -- No FK to allow logging for deleted entities
                        action TEXT NOT NULL,
                        changes JSONB,
                        performed_by TEXT NOT NULL,
                        performed_at TIMESTAMPTZ DEFAULT NOW(),
                        version INTEGER
                    );
                """)
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_entity_id ON entity_audit_log(entity_id);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_log_action ON entity_audit_log(action);")

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS relationships (
                        relationship_id TEXT PRIMARY KEY,
                        source_entity_id TEXT REFERENCES entities(entity_id) ON DELETE CASCADE,
                        target_entity_id TEXT REFERENCES entities(entity_id) ON DELETE CASCADE,
                        relationship_type TEXT NOT NULL,
                        attributes JSONB DEFAULT '{}'::jsonb,
                        confidence_score REAL DEFAULT 1.0,
                        source_document TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        created_by TEXT NOT NULL
                    );
                    """
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relationship_type ON relationships(relationship_type);"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relationship_src_tgt ON relationships(source_entity_id, target_entity_id);"
                )

                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS workflows (
                        workflow_id TEXT PRIMARY KEY,
                        state TEXT NOT NULL,
                        documents JSONB DEFAULT '{}'::jsonb,
                        agent_contexts JSONB DEFAULT '{}'::jsonb,
                        shared_data JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        created_by TEXT NOT NULL,
                        total_processing_time REAL DEFAULT 0.0,
                        completion_percentage REAL DEFAULT 0.0
                    );
                """)
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_workflows_state_updated ON workflows(state, updated_at DESC);")

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_path TEXT UNIQUE,
                        file_size INTEGER,
                        file_type TEXT,
                        file_hash TEXT UNIQUE,
                        processing_status TEXT DEFAULT 'pending',
                        processed_at TIMESTAMPTZ,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        source TEXT,
                        tags TEXT,
                        custom_metadata JSONB DEFAULT '{}'::jsonb
                    );
                    """
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_doc_meta_status ON documents(processing_status);"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_doc_meta_file_type ON documents(file_type);"
                )
                self.logger.info("Database schema created/verified successfully.")
        except asyncpg.PostgresError as e:
            self.logger.error("Database error during schema creation.", exception=e)
            raise DatabaseError("Failed to create database schema.", database_type="postgresql", cause=e)
        except Exception as e:
            self.logger.error("Unexpected error during schema creation.", exception=e)
            raise DatabaseError("Unexpected error creating schema.", cause=e)

    @detailed_log_function(LogCategory.DATABASE)
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all persistence components."""
        if not self.initialized:
            return {"status": "uninitialized", "manager": "EnhancedPersistenceManager"}
        
        self.logger.debug("Performing persistence health check.")
        pg_status = "unavailable"
        redis_status = "unavailable"
        cache_info = {}

        if self.connection_pool.pg_pool:
            try:
                async with self.connection_pool.get_pg_connection() as conn:
                    await conn.fetchval("SELECT 1")
                pg_status = "healthy"
            except Exception as e:
                pg_status = f"unhealthy: {str(e)}"
                self.logger.warning("PostgreSQL health check failed.", exception=e)
        
        if self.connection_pool.redis_pool:
            try:
                cache_info = await self.cache_manager.get_cache_stats()
                # Check if redis_version is present, indicating successful connection
                redis_status = "healthy" if cache_info.get("redis_version") else "unhealthy: Could not retrieve info"
            except Exception as e:
                redis_status = f"unhealthy: {str(e)}"
                self.logger.warning("Redis health check failed.", exception=e)
        
        overall_status = "healthy"
        if pg_status != "healthy" and self.connection_pool.database_url : # If PG is configured but unhealthy
            overall_status = "degraded"
        if redis_status != "healthy" and self.connection_pool.redis_url: # If Redis is configured but unhealthy
            overall_status = "degraded"
        if overall_status == "degraded" and (pg_status != "healthy" and redis_status != "healthy" and self.connection_pool.database_url and self.connection_pool.redis_url):
            overall_status = "unhealthy" # Both configured and unhealthy
        
        health_report = {
            "overall_status": overall_status,
            "postgresql_status": pg_status,
            "redis_status": redis_status,
            "redis_cache_info": cache_info,
            "connection_pool_stats": { # Basic pool stats
                "pg_pool_current_size": self.connection_pool.pg_pool.get_size() if self.connection_pool.pg_pool else 0,
                "pg_pool_free_size": self.connection_pool.pg_pool.get_freeprocs() if self.connection_pool.pg_pool else 0,
                # Redis pool size is harder to get directly from aioredis.ConnectionPool, often managed internally.
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        if self.metrics:
            try:
                self.metrics.update_pool_metrics(
                    pg_in_use=health_report["connection_pool_stats"]["pg_pool_current_size"],
                    pg_free=health_report["connection_pool_stats"]["pg_pool_free_size"],
                    redis_in_use=cache_info.get("connected_clients", 0),
                )
            except Exception:
                pass
        self.logger.info("Persistence health check complete.", parameters=health_report)
        return health_report
    
    @detailed_log_function(LogCategory.DATABASE)
    async def close(self):
        """Close all connections and clean up."""
        self.logger.info("Closing EnhancedPersistenceManager.")
        await self.connection_pool.close()
        self.initialized = False
        self.logger.info("EnhancedPersistenceManager closed.")

    # For service container compatibility
    async def initialize_service(self): # Renamed to avoid conflict with internal initialize
        await self.initialize()
    async def get_service_status(self) -> Dict[str, Any]:
        return await self.health_check()


# Factory function for service container

def create_enhanced_persistence_manager(
    *,
    config: Optional[Dict[str, Any]] = None,
    connection_pool: Optional[ConnectionPool] = None,
    metrics_exporter: Optional[Any] = None,
) -> EnhancedPersistenceManager:
    """Convenience factory for creating a persistence manager."""

    cfg = config or {}
    if connection_pool is None:
        connection_pool = ConnectionPool(
            cfg.get("database_url"),
            cfg.get("redis_url"),
            cfg.get("min_pg_connections", 5),
            cfg.get("max_pg_connections", 20),
            cfg.get("max_redis_connections", 10),
        )

    return EnhancedPersistenceManager(
        config=cfg,
        connection_pool=connection_pool,
        metrics_exporter=metrics_exporter,
    )
