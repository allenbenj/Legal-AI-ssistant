"""Knowledge Graph Manager - Centralized Knowledge Graph Operations.

Unified manager for knowledge graph operations providing clean integration
of Neo4j, entity management, and relationship handling.

This module provides a comprehensive knowledge graph management system
specifically designed for legal document processing and analysis.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import asyncio
from pathlib import Path
from enum import Enum
import asyncio
import hashlib

from ..core.enhanced_persistence import (
    ConnectionPool,
    EnhancedPersistenceManager,
    EntityRecord,
    RelationshipRecord,
    EntityStatus,
)

# Import detailed logging
from ..core.detailed_logging import get_detailed_logger, LogCategory, detailed_log_function

# Initialize loggers
kg_logger = get_detailed_logger("Knowledge_Graph_Manager", LogCategory.KNOWLEDGE_GRAPH)
entity_logger = get_detailed_logger("Entity_Management", LogCategory.KNOWLEDGE_GRAPH)
relationship_logger = get_detailed_logger("Relationship_Management", LogCategory.KNOWLEDGE_GRAPH)
query_logger = get_detailed_logger("Graph_Queries", LogCategory.KNOWLEDGE_GRAPH)

class EntityType(Enum):
    """Supported entity types in the knowledge graph.
    
    Defines all legal entity types that can be represented in the
    knowledge graph for comprehensive legal document analysis.
    """
    PERSON = "person"
    ORGANIZATION = "organization"
    CASE = "case"
    DOCUMENT = "document"
    STATUTE = "statute"
    REGULATION = "regulation"
    EVIDENCE = "evidence"
    JUDGE = "judge"
    LAWYER = "lawyer"
    COURT = "court"
    DATE = "date"
    LOCATION = "location"
    CONCEPT = "concept"

class RelationshipType(Enum):
    """Supported relationship types in the knowledge graph.
    
    Defines all relationship types that can exist between entities
    in the legal knowledge graph, enabling complex query patterns.
    """
    FILED_BY = "filed_by"
    REPRESENTS = "represents"
    CITED_IN = "cited_in"
    REFERENCES = "references"
    INVOLVES = "involves"
    DECIDED_BY = "decided_by"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    AMENDS = "amends"
    RELATES_TO = "relates_to"
    PART_OF = "part_of"
    LOCATED_IN = "located_in"
    OCCURRED_ON = "occurred_on"

@dataclass
class Entity:
    """Knowledge graph entity representation."""
    id: str
    type: EntityType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_document: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Relationship:
    """Knowledge graph relationship representation."""
    id: str
    source_entity_id: str
    target_entity_id: str
    type: RelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_document: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class GraphQuery:
    """Knowledge graph query representation."""
    query_type: str
    parameters: Dict[str, Any]
    limit: int = 100
    offset: int = 0

@dataclass
class QueryResult:
    """Knowledge graph query result."""
    entities: List[Entity]
    relationships: List[Relationship]
    total_count: int
    execution_time: float
    query_metadata: Dict[str, Any] = field(default_factory=dict)

class KnowledgeGraphManager:
    """
    Centralized manager for knowledge graph operations.
    
    Features:
    - Entity management (create, update, delete, query)
    - Relationship management with type validation
    - Graph traversal and path finding
    - Query optimization and caching
    - Batch operations for performance
    - Integration with external graph databases (Neo4j)
    """
    
    @detailed_log_function(LogCategory.KNOWLEDGE_GRAPH)
    def __init__(
        self,
        storage_dir: str = "./storage/knowledge_graph",
        service_config: Optional[Dict[str, Any]] = None,
        connection_pool: ConnectionPool | None = None,
    ) -> None:
        """Initialize knowledge graph manager."""
        kg_logger.info("=== INITIALIZING KNOWLEDGE GRAPH MANAGER ===")

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.config = service_config or {}
        self.connection_pool = connection_pool
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.entity_index: Dict[EntityType, List[str]] = {et: [] for et in EntityType}
        self.relationship_index: Dict[RelationshipType, List[str]] = {rt: [] for rt in RelationshipType}
        
        # Neo4j integration (optional)
        self.neo4j_driver = None
        self.neo4j_enabled = self.config.get('enable_neo4j', False)
        
        # Performance tracking
        self.query_cache: Dict[str, QueryResult] = {}
        self.cache_ttl = self.config.get('cache_ttl_seconds', 3600)
        self.cache_manager = cache_manager
        self.metrics = metrics_exporter
        self.query_history: List[Dict[str, Any]] = []
        
        if self.neo4j_enabled:
            self._init_neo4j()

        # Initialize persistence layer using the shared connection pool if provided
        self._init_persistence()
        
        kg_logger.info("Knowledge graph manager initialization complete", parameters={
            'storage_dir': str(self.storage_dir),
            'neo4j_enabled': self.neo4j_enabled,
            'cache_ttl': self.cache_ttl
        })
    
    @detailed_log_function(LogCategory.KNOWLEDGE_GRAPH)
    def _init_persistence(self):
        """Initialize database persistence layer."""
        persistence_cfg = self.config.get("persistence", {})
        if not self.persistence:
            if self.connection_pool:
                self.persistence = EnhancedPersistenceManager(
                    connection_pool=self.connection_pool,
                    config=persistence_cfg,
                )
            else:
                self.persistence = EnhancedPersistenceManager(
                    persistence_cfg.get("database_url"),
                    persistence_cfg.get("redis_url"),
                    config=persistence_cfg,
                )
        if self.persistence:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.persistence.initialize())
                else:
                    loop.run_until_complete(self.persistence.initialize())
            except RuntimeError:
                asyncio.run(self.persistence.initialize())
    
    @detailed_log_function(LogCategory.KNOWLEDGE_GRAPH)
    def _init_neo4j(self):
        """Initialize Neo4j connection if enabled."""
        try:
            # Import Neo4j driver (optional dependency)
            from neo4j import GraphDatabase
            
            neo4j_config = self.config.get('neo4j', {})
            uri = neo4j_config.get('uri', 'neo4j://localhost:7687')
            username = neo4j_config.get('username', 'neo4j')
            password = neo4j_config.get('password', 'password')
            
            self.neo4j_driver = GraphDatabase.driver(uri, auth=(username, password))
            
            # Test connection
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")
            
            kg_logger.info("Neo4j connection established", parameters={
                'uri': uri,
                'username': username
            })
            
        except ImportError:
            kg_logger.warning("Neo4j driver not available - using local storage only")
            self.neo4j_enabled = False
        except Exception as e:
            kg_logger.error("Failed to connect to Neo4j", exception=e)
            self.neo4j_enabled = False
    
    # ==================== ENTITY OPERATIONS ====================
    
    @detailed_log_function(LogCategory.KNOWLEDGE_GRAPH)
    async def create_entity(self, entity_type: EntityType, name: str, 
                           properties: Optional[Dict[str, Any]] = None,
                           confidence: float = 1.0,
                           source_document: Optional[str] = None) -> Entity:
        """Create a new entity in the knowledge graph."""
        entity_logger.info(f"Creating entity: {name} ({entity_type.value})")
        
        async with self._lock:
            # Generate unique ID
            entity_id = self._generate_entity_id(entity_type, name)
            
            # Check if entity already exists
            if entity_id in self.entities:
                existing = self.entities[entity_id]
                entity_logger.info(f"Entity already exists: {name}", parameters={
                    'existing_id': existing.id,
                    'existing_confidence': existing.confidence
                })
                
                # Update confidence if higher
                if confidence > existing.confidence:
                    existing.confidence = confidence
                    existing.updated_at = datetime.now()
                    if source_document:
                        existing.source_document = source_document
                    if self.persistence:
                        await self.persistence.entity_repo.update_entity(
                            existing.id,
                            {"confidence_score": confidence},
                            "kg_manager",
                        )
                
                return existing
            
            # Create new entity
            entity = Entity(
                id=entity_id,
                type=entity_type,
                name=name,
                properties=properties or {},
                confidence=confidence,
                source_document=source_document
            )
            
            if self.persistence:
                record = EntityRecord(
                    entity_id=entity_id,
                    entity_type=entity_type.value,
                    canonical_name=name,
                    attributes=properties or {},
                    confidence_score=confidence,
                    status=EntityStatus.ACTIVE,
                    created_by="kg_manager",
                    updated_by="kg_manager",
                    source_documents=[source_document] if source_document else [],
                )
                await self.persistence.entity_repo.create_entity(record)

            self.entities[entity_id] = entity
            self.entity_index[entity_type].append(entity_id)
            
            # Sync to Neo4j if enabled
            if self.neo4j_enabled:
                await self._sync_entity_to_neo4j(entity)
            
            entity_logger.info(f"Entity created successfully: {name}", parameters={
                'entity_id': entity_id,
                'type': entity_type.value,
                'confidence': confidence
            })
            
            return entity
    
    @detailed_log_function(LogCategory.KNOWLEDGE_GRAPH)
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
    
    @detailed_log_function(LogCategory.KNOWLEDGE_GRAPH)
    async def find_entities(self, entity_type: Optional[EntityType] = None,
                           name_pattern: Optional[str] = None,
                           properties_filter: Optional[Dict[str, Any]] = None,
                           limit: int = 100) -> List[Entity]:
        """Find entities matching criteria."""
        query_logger.trace("Finding entities", parameters={
            'entity_type': entity_type.value if entity_type else None,
            'name_pattern': name_pattern,
            'limit': limit
        })

        cache_key = None
        if self.cache_manager:
            base = {
                "type": entity_type.value if entity_type else None,
                "name": name_pattern,
                "props": properties_filter,
                "limit": limit,
            }
            cache_key = "kg_query:" + hashlib.sha256(str(base).encode()).hexdigest()
            cached = await self.cache_manager.get(cache_key)
            if cached:
                if self.metrics:
                    self.metrics.inc_kg_query(cache_hit=True)
                return [Entity(**c) for c in cached]

        if self.metrics:
            self.metrics.inc_kg_query()
        
            entities = []

            # Filter by type
            entity_ids = (
                self.entity_index[entity_type]
                if entity_type
                else list(self.entities.keys())
            )

            for entity_id in entity_ids:
                entity = self.entities[entity_id]

                # Filter by name pattern
                if name_pattern and name_pattern.lower() not in entity.name.lower():
                    continue

                if properties_filter:
                    match = True
                    for key, value in properties_filter.items():
                        if key not in entity.properties or entity.properties[key] != value:
                            match = False
                            break
                    if not match:
                        continue

                entities.append(entity)

                if len(entities) >= limit:
                    break

            query_logger.info(f"Found {len(entities)} entities")
            if cache_key and self.cache_manager:
                try:
                    await self.cache_manager.set(
                        cache_key,
                        [asdict(e) for e in entities],
                        ttl_seconds=self.cache_ttl,
                    )
                except Exception:
                    pass
            return entities
    
    # ==================== RELATIONSHIP OPERATIONS ====================
    
    @detailed_log_function(LogCategory.KNOWLEDGE_GRAPH)
    async def create_relationship(self, source_entity_id: str, target_entity_id: str,
                                 relationship_type: RelationshipType,
                                 properties: Optional[Dict[str, Any]] = None,
                                 confidence: float = 1.0,
                                 source_document: Optional[str] = None) -> Relationship:
        """Create a new relationship in the knowledge graph."""
        relationship_logger.info(f"Creating relationship: {source_entity_id} -{relationship_type.value}-> {target_entity_id}")
        
        async with self._lock:
            # Validate entities exist
            if source_entity_id not in self.entities:
                raise ValueError(f"Source entity not found: {source_entity_id}")
            if target_entity_id not in self.entities:
                raise ValueError(f"Target entity not found: {target_entity_id}")
            
            # Generate unique ID
            rel_id = self._generate_relationship_id(source_entity_id, target_entity_id, relationship_type)
            
            # Check if relationship already exists
            if rel_id in self.relationships:
                existing = self.relationships[rel_id]
                relationship_logger.info(f"Relationship already exists", parameters={
                    'existing_id': existing.id,
                    'existing_confidence': existing.confidence
                })
                
                # Update confidence if higher
                if confidence > existing.confidence:
                    existing.confidence = confidence
                    if properties:
                        existing.properties.update(properties)
                    if source_document:
                        existing.source_document = source_document
                    if self.persistence:
                        await self.persistence.relationship_repo.create_relationship(
                            RelationshipRecord(
                                relationship_id=rel_id,
                                source_entity_id=source_entity_id,
                                target_entity_id=target_entity_id,
                                relationship_type=relationship_type.value,
                                attributes=existing.properties,
                                confidence_score=confidence,
                                source_document=source_document,
                                created_by="kg_manager",
                            )
                        )
                
                return existing
            
            # Create new relationship
            relationship = Relationship(
                id=rel_id,
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                type=relationship_type,
                properties=properties or {},
                confidence=confidence,
                source_document=source_document
            )
            
            if self.persistence:
                record = RelationshipRecord(
                    relationship_id=rel_id,
                    source_entity_id=source_entity_id,
                    target_entity_id=target_entity_id,
                    relationship_type=relationship_type.value,
                    attributes=properties or {},
                    confidence_score=confidence,
                    source_document=source_document,
                    created_by="kg_manager",
                )
                await self.persistence.relationship_repo.create_relationship(record)

            self.relationships[rel_id] = relationship
            self.relationship_index[relationship_type].append(rel_id)
            
            # Sync to Neo4j if enabled
            if self.neo4j_enabled:
                await self._sync_relationship_to_neo4j(relationship)
            
            relationship_logger.info(f"Relationship created successfully", parameters={
                'relationship_id': rel_id,
                'type': relationship_type.value,
                'confidence': confidence
            })
            
            return relationship
    
    # ==================== GRAPH QUERIES ====================
    
    @detailed_log_function(LogCategory.KNOWLEDGE_GRAPH)
    async def find_connected_entities(self, entity_id: str,
                                     relationship_types: Optional[List[RelationshipType]] = None,
                                     max_depth: int = 2) -> List[Entity]:
        """Find entities connected to the given entity."""
        connected_entities: set[str] = set()

        if self.persistence:
            records = await self.persistence.relationship_repo.get_relationships_by_entity(entity_id)
            for rec in records:
                rel_type = RelationshipType(rec.relationship_type)
                if relationship_types and rel_type not in relationship_types:
                    continue
                target_id = rec.target_entity_id if rec.source_entity_id == entity_id else rec.source_entity_id
                if target_id != entity_id:
                    connected_entities.add(target_id)
        else:
            with self._lock:
                for rel in self.relationships.values():
                    if relationship_types and rel.type not in relationship_types:
                        continue
                    target_id = (
                        rel.target_entity_id if rel.source_entity_id == entity_id else rel.source_entity_id
                    )
                    if target_id != entity_id:
                        connected_entities.add(target_id)

        return [self.entities[eid] for eid in connected_entities if eid in self.entities]
    
    # ==================== UTILITY METHODS ====================
    
    def _generate_entity_id(self, entity_type: EntityType, name: str) -> str:
        """Generate unique entity ID."""
        content = f"{entity_type.value}:{name.lower()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_relationship_id(self, source_id: str, target_id: str, rel_type: RelationshipType) -> str:
        """Generate unique relationship ID."""
        content = f"{source_id}:{rel_type.value}:{target_id}"
        return hashlib.md5(content.encode()).hexdigest()

    def _entity_from_record(self, record: EntityRecord) -> Entity:
        return Entity(
            id=record.entity_id,
            type=EntityType(record.entity_type),
            name=record.canonical_name,
            properties=record.attributes,
            confidence=record.confidence_score,
            source_document=record.source_documents[0] if record.source_documents else None,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
    

    
    async def _sync_entity_to_neo4j(self, entity: Entity):
        """Sync entity to Neo4j database."""
        if not self.neo4j_enabled or not self.neo4j_driver:
            return
        
        try:
            with self.neo4j_driver.session() as session:
                # Create or update entity
                query = """
                MERGE (e:Entity {id: $id})
                SET e.type = $type,
                    e.name = $name,
                    e.confidence = $confidence,
                    e.updated_at = $updated_at
                """
                session.run(query, {
                    'id': entity.id,
                    'type': entity.type.value,
                    'name': entity.name,
                    'confidence': entity.confidence,
                    'updated_at': entity.updated_at.isoformat()
                })
                
        except Exception as e:
            kg_logger.error(f"Failed to sync entity to Neo4j: {entity.id}", exception=e)
    
    async def _sync_relationship_to_neo4j(self, relationship: Relationship):
        """Sync relationship to Neo4j database."""
        if not self.neo4j_enabled or not self.neo4j_driver:
            return
        
        try:
            with self.neo4j_driver.session() as session:
                # Create relationship
                query = f"""
                MATCH (source:Entity {{id: $source_id}})
                MATCH (target:Entity {{id: $target_id}})
                MERGE (source)-[r:{relationship.type.value.upper()}]->(target)
                SET r.confidence = $confidence,
                    r.created_at = $created_at
                """
                session.run(query, {
                    'source_id': relationship.source_entity_id,
                    'target_id': relationship.target_entity_id,
                    'confidence': relationship.confidence,
                    'created_at': relationship.created_at.isoformat()
                })
                
        except Exception as e:
            kg_logger.error(f"Failed to sync relationship to Neo4j: {relationship.id}", exception=e)
    
    @detailed_log_function(LogCategory.KNOWLEDGE_GRAPH)
    async def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        async with self._lock:
            entity_counts = {et.value: len(ids) for et, ids in self.entity_index.items() if ids}
            relationship_counts = {rt.value: len(ids) for rt, ids in self.relationship_index.items() if ids}
            
            stats = {
                'total_entities': len(self.entities),
                'total_relationships': len(self.relationships),
                'entity_types': entity_counts,
                'relationship_types': relationship_counts,
                'neo4j_enabled': self.neo4j_enabled,
                'storage_dir': str(self.storage_dir)
            }
            
            kg_logger.info("Knowledge graph statistics generated", parameters=stats)
            return stats
    
    async def initialize(self):
        """Async initialization for service container compatibility."""
        kg_logger.info("Knowledge graph manager async initialization complete")
        return self
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for service container monitoring."""
        health = {
            'healthy': True,
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'neo4j_enabled': self.neo4j_enabled,
            'storage_accessible': self.storage_dir.exists()
        }
        
        # Test Neo4j connection if enabled
        if self.neo4j_enabled and self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    session.run("RETURN 1")
                health['neo4j_connection'] = True
            except Exception:
                health['healthy'] = False
                health['neo4j_connection'] = False
        
        return health

# Service container factory function
def create_knowledge_graph_manager(
    service_container: "ServiceContainer",
    *,
    connection_pool: ConnectionPool,
    config: Optional[Dict[str, Any]] = None,
) -> KnowledgeGraphManager:
    """Factory for :class:`ServiceContainer`"""
    return KnowledgeGraphManager(
        storage_dir=config.get("STORAGE_DIR", "./storage/knowledge_graph") if config else "./storage/knowledge_graph",
        service_config=config,
        connection_pool=connection_pool,
    )
