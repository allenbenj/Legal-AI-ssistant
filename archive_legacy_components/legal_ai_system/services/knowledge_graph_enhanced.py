"""Enhanced Knowledge Graph Builder - Extracted from delete_when_done/knowledge_graph_builder.py

Key features extracted:
- Dual storage architecture (NetworkX in-memory + Neo4j persistence)
- Entity and relationship management with validation
- JSON data loading and synchronization capabilities
- Legal entity modeling (Person, Party, Case, Evidence, etc.)
- Comprehensive graph validation and error checking

This provides a robust foundation for legal knowledge graph operations.
"""

import networkx as nx
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Enhanced entity representation"""
    entity_type: str
    entity_id: str
    attributes: Dict[str, Any]
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass 
class Relationship:
    """Enhanced relationship representation"""
    from_id: str
    to_id: str
    relationship_type: str
    attributes: Dict[str, Any] = None
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at


class EnhancedKnowledgeGraphBuilder:
    """Enhanced knowledge graph builder with dual storage and validation"""
    
    def __init__(self, neo4j_uri: Optional[str] = None, 
                 neo4j_user: Optional[str] = None, 
                 neo4j_password: Optional[str] = None,
                 enable_neo4j: bool = True):
        """
        Initialize knowledge graph builder
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            enable_neo4j: Whether to enable Neo4j integration
        """
        # NetworkX graph for in-memory operations
        self.graph = nx.DiGraph()
        
        # Neo4j integration (optional)
        self.driver = None
        self.neo4j_enabled = enable_neo4j
        
        if enable_neo4j and all([neo4j_uri, neo4j_user, neo4j_password]):
            try:
                from neo4j import GraphDatabase
                self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
                logger.info("Neo4j driver initialized successfully")
            except ImportError:
                logger.warning("Neo4j driver not available. Install with: pip install neo4j")
                self.neo4j_enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j driver: {e}")
                self.neo4j_enabled = False
        else:
            logger.info("Neo4j integration disabled or credentials not provided")
            self.neo4j_enabled = False
        
        # Validation rules and statistics
        self.validation_errors = []
        self.entity_count = 0
        self.relationship_count = 0
    
    def add_entity(self, entity: Entity) -> bool:
        """
        Add entity to knowledge graph
        
        Args:
            entity: Entity object to add
            
        Returns:
            bool: True if entity was added successfully
        """
        try:
            logger.debug(f"Adding entity: {entity.entity_type}, ID: {entity.entity_id}")
            
            # Validate entity data
            if not entity.entity_type or not entity.entity_id:
                raise ValueError("Entity type and ID are required")
            
            # Add to NetworkX graph
            self.graph.add_node(
                entity.entity_id, 
                entity_type=entity.entity_type,
                created_at=entity.created_at,
                updated_at=entity.updated_at,
                **entity.attributes
            )
            
            # Add to Neo4j if enabled
            if self.neo4j_enabled and self.driver:
                self._add_entity_to_neo4j(entity)
            
            self.entity_count += 1
            logger.info(f"Successfully added entity: {entity.entity_type}:{entity.entity_id}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to add entity {entity.entity_type}:{entity.entity_id}: {e}"
            logger.error(error_msg)
            self.validation_errors.append(error_msg)
            return False
    
    def add_relationship(self, relationship: Relationship) -> bool:
        """
        Add relationship to knowledge graph
        
        Args:
            relationship: Relationship object to add
            
        Returns:
            bool: True if relationship was added successfully
        """
        try:
            logger.debug(f"Adding relationship: {relationship.relationship_type} from {relationship.from_id} to {relationship.to_id}")
            
            # Validate relationship data
            if not all([relationship.from_id, relationship.to_id, relationship.relationship_type]):
                raise ValueError("From ID, To ID, and relationship type are required")
            
            # Check if entities exist
            if relationship.from_id not in self.graph.nodes:
                raise ValueError(f"Source entity {relationship.from_id} not found in graph")
            if relationship.to_id not in self.graph.nodes:
                raise ValueError(f"Target entity {relationship.to_id} not found in graph")
            
            # Add to NetworkX graph
            self.graph.add_edge(
                relationship.from_id, 
                relationship.to_id, 
                relationship_type=relationship.relationship_type,
                created_at=relationship.created_at,
                updated_at=relationship.updated_at,
                **relationship.attributes
            )
            
            # Add to Neo4j if enabled
            if self.neo4j_enabled and self.driver:
                self._add_relationship_to_neo4j(relationship)
            
            self.relationship_count += 1
            logger.info(f"Successfully added relationship: {relationship.from_id} --{relationship.relationship_type}--> {relationship.to_id}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to add relationship {relationship.from_id}--{relationship.relationship_type}-->{relationship.to_id}: {e}"
            logger.error(error_msg)
            self.validation_errors.append(error_msg)
            return False
    
    def _add_entity_to_neo4j(self, entity: Entity):
        """Add entity to Neo4j database"""
        with self.driver.session() as session:
            # Sanitize entity type for Cypher (remove spaces, special chars)
            safe_type = entity.entity_type.replace(" ", "_").replace("-", "_")
            
            query = f"MERGE (n:{safe_type} {{id: $entity_id}}) SET n += $attributes, n.created_at = $created_at, n.updated_at = $updated_at"
            
            session.run(
                query, 
                entity_id=entity.entity_id, 
                attributes=entity.attributes,
                created_at=entity.created_at,
                updated_at=entity.updated_at
            )
    
    def _add_relationship_to_neo4j(self, relationship: Relationship):
        """Add relationship to Neo4j database"""
        with self.driver.session() as session:
            # Sanitize relationship type for Cypher
            safe_type = relationship.relationship_type.replace(" ", "_").replace("-", "_")
            
            query = (
                "MATCH (a {id: $from_id}), (b {id: $to_id}) "
                f"MERGE (a)-[r:{safe_type}]->(b) "
                "SET r += $attributes, r.created_at = $created_at, r.updated_at = $updated_at"
            )
            
            session.run(
                query,
                from_id=relationship.from_id,
                to_id=relationship.to_id,
                attributes=relationship.attributes,
                created_at=relationship.created_at,
                updated_at=relationship.updated_at
            )
    
    def find_entities(self, entity_type: Optional[str] = None, 
                     attributes: Optional[Dict[str, Any]] = None,
                     limit: int = 100) -> List[Entity]:
        """Find entities by type and/or attributes"""
        results = []
        
        for node, data in self.graph.nodes(data=True):
            # Filter by entity type
            if entity_type and data.get('entity_type') != entity_type:
                continue
                
            # Filter by attributes
            if attributes:
                match = True
                for key, value in attributes.items():
                    if data.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Create entity object
            entity_attrs = {k: v for k, v in data.items() 
                          if k not in ['entity_type', 'created_at', 'updated_at']}
            
            entity = Entity(
                entity_type=data.get('entity_type', 'Unknown'),
                entity_id=node,
                attributes=entity_attrs,
                created_at=data.get('created_at'),
                updated_at=data.get('updated_at')
            )
            results.append(entity)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_relationships(self, entity_id: Optional[str] = None,
                         relationship_type: Optional[str] = None,
                         limit: int = 100) -> List[Relationship]:
        """Get relationships for an entity or by type"""
        results = []
        
        edges = self.graph.edges(data=True)
        if entity_id:
            # Get edges involving the specific entity
            edges = [(u, v, d) for u, v, d in edges if u == entity_id or v == entity_id]
        
        for u, v, data in edges:
            # Filter by relationship type
            if relationship_type and data.get('relationship_type') != relationship_type:
                continue
            
            # Create relationship object
            rel_attrs = {k: v for k, v in data.items() 
                        if k not in ['relationship_type', 'created_at', 'updated_at']}
            
            relationship = Relationship(
                from_id=u,
                to_id=v,
                relationship_type=data.get('relationship_type', 'Unknown'),
                attributes=rel_attrs,
                created_at=data.get('created_at'),
                updated_at=data.get('updated_at')
            )
            results.append(relationship)
            
            if len(results) >= limit:
                break
        
        return results
    
    def find_connected_entities(self, entity_id: str, 
                               relationship_types: Optional[List[str]] = None,
                               max_depth: int = 2) -> List[Dict[str, Any]]:
        """Find entities connected to a given entity"""
        if entity_id not in self.graph.nodes:
            return []
        
        connected = []
        visited = set([entity_id])
        
        def explore(current_id: str, depth: int):
            if depth >= max_depth:
                return
            
            # Get neighbors (both incoming and outgoing)
            neighbors = list(self.graph.successors(current_id)) + list(self.graph.predecessors(current_id))
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                # Check relationship type filter
                edge_data = None
                if self.graph.has_edge(current_id, neighbor):
                    edge_data = self.graph.edges[current_id, neighbor]
                elif self.graph.has_edge(neighbor, current_id):
                    edge_data = self.graph.edges[neighbor, current_id]
                
                if relationship_types and edge_data:
                    rel_type = edge_data.get('relationship_type')
                    if rel_type not in relationship_types:
                        continue
                
                visited.add(neighbor)
                node_data = self.graph.nodes[neighbor]
                
                connected.append({
                    'entity_id': neighbor,
                    'entity_type': node_data.get('entity_type', 'Unknown'),
                    'attributes': {k: v for k, v in node_data.items() 
                                 if k not in ['entity_type']},
                    'depth': depth + 1,
                    'path_length': depth + 1
                })
                
                # Recursively explore
                explore(neighbor, depth + 1)
        
        explore(entity_id, 0)
        return connected
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        entity_types = {}
        relationship_types = {}
        
        # Count entity types
        for node, data in self.graph.nodes(data=True):
            entity_type = data.get('entity_type', 'Unknown')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Count relationship types
        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get('relationship_type', 'Unknown')
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        return {
            "total_entities": self.graph.number_of_nodes(),
            "total_relationships": self.graph.number_of_edges(),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "isolated_nodes": len(list(nx.isolates(self.graph))),
            "connected_components": nx.number_weakly_connected_components(self.graph),
            "neo4j_enabled": self.neo4j_enabled
        }
    
    def export_graph(self, format: str = "json") -> Dict[str, Any]:
        """Export the entire graph"""
        entities = []
        relationships = []
        
        # Export entities
        for node, data in self.graph.nodes(data=True):
            entity = {
                "id": node,
                "type": data.get('entity_type', 'Unknown'),
                "attributes": {k: v for k, v in data.items() 
                             if k not in ['entity_type']}
            }
            entities.append(entity)
        
        # Export relationships
        for u, v, data in self.graph.edges(data=True):
            relationship = {
                "from_id": u,
                "to_id": v,
                "type": data.get('relationship_type', 'RELATED_TO'),
                "attributes": {k: v for k, v in data.items() 
                             if k not in ['relationship_type']}
            }
            relationships.append(relationship)
        
        return {
            "entities": entities,
            "relationships": relationships,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "entity_count": len(entities),
                "relationship_count": len(relationships),
                "format": format
            }
        }
    
    def sync_to_neo4j(self, uri: str, user: str, password: str) -> bool:
        """Synchronize entire NetworkX graph to Neo4j"""
        try:
            from neo4j import GraphDatabase
            
            with GraphDatabase.driver(uri, auth=(user, password)) as driver:
                with driver.session() as session:
                    # Sync nodes
                    for node, data in self.graph.nodes(data=True):
                        entity_type = data.get('entity_type', 'Unknown')
                        safe_type = entity_type.replace(" ", "_").replace("-", "_")
                        
                        query = f"MERGE (n:{safe_type} {{id: $node_id}}) SET n += $attributes"
                        session.run(query, node_id=node, attributes=data)
                    
                    # Sync edges
                    for u, v, data in self.graph.edges(data=True):
                        rel_type = data.get('relationship_type', 'RELATED_TO')
                        safe_type = rel_type.replace(" ", "_").replace("-", "_")
                        
                        query = (
                            "MATCH (a {id: $from_id}), (b {id: $to_id}) "
                            f"MERGE (a)-[r:{safe_type}]->(b) SET r += $attributes"
                        )
                        session.run(query, from_id=u, to_id=v, attributes=data)
            
            logger.info("Neo4j synchronization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync to Neo4j: {e}")
            return False
    
    def clear_graph(self) -> None:
        """Clear all data from the graph"""
        self.graph.clear()
        self.entity_count = 0
        self.relationship_count = 0
        self.validation_errors.clear()
        logger.info("Graph cleared successfully")
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        if entity_id in self.graph.nodes:
            data = self.graph.nodes[entity_id]
            entity_attrs = {k: v for k, v in data.items() 
                          if k not in ['entity_type', 'created_at', 'updated_at']}
            
            return Entity(
                entity_type=data.get('entity_type', 'Unknown'),
                entity_id=entity_id,
                attributes=entity_attrs,
                created_at=data.get('created_at'),
                updated_at=data.get('updated_at')
            )
        return None
    
    def close(self):
        """Close connections and cleanup"""
        if self.driver:
            logger.info("Closing Neo4j driver connection")
            self.driver.close()