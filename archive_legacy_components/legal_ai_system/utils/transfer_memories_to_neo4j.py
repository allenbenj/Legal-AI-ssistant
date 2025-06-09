#!/usr/bin/env python3
"""
Transfer Claude Code memories to Neo4j database
Transfers entities, relationships, and observations from MCP memory to Neo4j
"""

import json
import logging
from typing import Dict, List, Any, Optional

try:
    from neo4j import GraphDatabase
except ImportError:
    print("neo4j package not installed. Please install with: pip install neo4j")
    exit(1)

# Neo4j connection details
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "CaseDBMS"

class MemoryTransfer:
    """Transfer memories from MCP memory system to Neo4j"""
    
    def __init__(self):
        self.driver = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI, 
                auth=(NEO4J_USER, NEO4J_PASSWORD),
                database=NEO4J_DATABASE
            )
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                self.logger.info(f"Connected to Neo4j successfully. Test value: {test_value}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def clear_claude_memories(self):
        """Clear existing Claude Code memories from database"""
        with self.driver.session() as session:
            # Delete all nodes and relationships with Claude prefix
            session.run("""
                MATCH (n) 
                WHERE n.source = 'claude_code' OR n.name STARTS WITH 'Claude'
                DETACH DELETE n
            """)
            self.logger.info("Cleared existing Claude Code memories")
    
    def create_entity(self, session, entity: Dict[str, Any]):
        """Create entity node in Neo4j"""
        name = entity.get('name', '')
        entity_type = entity.get('entityType', 'unknown')
        observations = entity.get('observations', [])
        
        # Create entity node
        session.run("""
            CREATE (e:Entity {
                name: $name,
                entity_type: $entity_type,
                source: 'claude_code',
                created_at: datetime(),
                observations: $observations
            })
        """, name=name, entity_type=entity_type, observations=observations)
        
        # Create observation nodes and relationships
        for i, observation in enumerate(observations):
            session.run("""
                MATCH (e:Entity {name: $name, source: 'claude_code'})
                CREATE (o:Observation {
                    id: $obs_id,
                    content: $content,
                    source: 'claude_code',
                    created_at: datetime()
                })
                CREATE (e)-[:HAS_OBSERVATION]->(o)
            """, 
            name=name, 
            obs_id=f"{name}_{i}", 
            content=observation
            )
    
    def create_relationship(self, session, relation: Dict[str, Any]):
        """Create relationship between entities"""
        from_entity = relation.get('from', '')
        to_entity = relation.get('to', '')
        relation_type = relation.get('relationType', 'RELATED_TO')
        
        # Clean relation type for Neo4j (no spaces, uppercase)
        relation_type_clean = relation_type.replace(' ', '_').upper()
        
        session.run(f"""
            MATCH (from:Entity {{name: $from_name, source: 'claude_code'}})
            MATCH (to:Entity {{name: $to_name, source: 'claude_code'}})
            CREATE (from)-[r:{relation_type_clean} {{
                original_type: $original_type,
                source: 'claude_code',
                created_at: datetime()
            }}]->(to)
        """, 
        from_name=from_entity, 
        to_name=to_entity,
        original_type=relation_type
        )
    
    def transfer_memories(self, memory_data: Dict[str, Any]):
        """Transfer all memories to Neo4j"""
        entities = memory_data.get('entities', [])
        relations = memory_data.get('relations', [])
        
        with self.driver.session() as session:
            # Create entities first
            self.logger.info(f"Creating {len(entities)} entities...")
            for entity in entities:
                try:
                    self.create_entity(session, entity)
                    self.logger.debug(f"Created entity: {entity.get('name', 'unknown')}")
                except Exception as e:
                    self.logger.error(f"Failed to create entity {entity.get('name', 'unknown')}: {e}")
            
            # Create relationships
            self.logger.info(f"Creating {len(relations)} relationships...")
            for relation in relations:
                try:
                    self.create_relationship(session, relation)
                    self.logger.debug(f"Created relationship: {relation.get('from', '')} -> {relation.get('to', '')}")
                except Exception as e:
                    self.logger.error(f"Failed to create relationship {relation.get('from', '')} -> {relation.get('to', '')}: {e}")
    
    def verify_transfer(self) -> Dict[str, int]:
        """Verify the transfer was successful"""
        with self.driver.session() as session:
            # Count entities
            entity_count = session.run("""
                MATCH (e:Entity {source: 'claude_code'}) 
                RETURN count(e) as count
            """).single()["count"]
            
            # Count observations
            obs_count = session.run("""
                MATCH (o:Observation {source: 'claude_code'}) 
                RETURN count(o) as count
            """).single()["count"]
            
            # Count relationships
            rel_count = session.run("""
                MATCH ()-[r {source: 'claude_code'}]->() 
                RETURN count(r) as count
            """).single()["count"]
            
            return {
                'entities': entity_count,
                'observations': obs_count,
                'relationships': rel_count
            }
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

def main():
    """Main transfer function"""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Memory data from Claude Code MCP memory
    memory_data = {
        "entities": [
            {
                "name": "Legal AI System Architecture",
                "entityType": "software_system",
                "observations": [
                    "Microservices-based legal document processing platform with service container pattern",
                    "Uses multi-LLM approach: xAI/Grok (primary), Ollama (fallback), OpenAI (optional)",
                    "Supports dynamic model switching based on task complexity (grok-3-mini for speed, grok-3-reasoning for complex analysis)",
                    "Core architecture 100% complete with service container, LLM providers, configuration management",
                    "Agent-based processing with BaseAgent framework providing async processing, error handling, retry logic"
                ]
            },
            {
                "name": "Document Processing System",
                "entityType": "component",
                "observations": [
                    "90% complete DocumentProcessorAgent supporting PDF, DOCX, TXT, MD, HTML, RTF",
                    "Structured data processing for Excel (XLSX/XLS), CSV with database schema generation",
                    "OCR integration with Tesseract for image-based documents",
                    "Legal document classification and type detection",
                    "Multi-format processing strategies: full processing, structured data, reference-only"
                ]
            },
            {
                "name": "Vector Storage System",
                "entityType": "component",
                "observations": [
                    "60% complete - multiple implementations exist but not integrated",
                    "ultimate_vector_store.py: Advanced FAISS with dual indexes",
                    "optimized_vector_store.py: Performance-optimized version",
                    "extracted_components/vector_store_enhanced.py: Production-ready implementation",
                    "Missing VectorStoreManager integration with service container"
                ]
            },
            {
                "name": "Knowledge Graph System",
                "entityType": "component",
                "observations": [
                    "70% complete - enhanced_graph_builder.py with NetworkX + Neo4j dual storage",
                    "realtime_graph_manager.py for high-performance operations",
                    "Supports 15 entity types and 24+ relationship types for legal domain",
                    "Missing service container integration and GUI visualization",
                    "Needs KnowledgeGraphManager wrapper class"
                ]
            },
            {
                "name": "Memory Management System",
                "entityType": "component",
                "observations": [
                    "50% complete - reviewable_memory.py provides human-in-the-loop validation",
                    "Missing context window management and auto-summarization",
                    "Lacks session persistence and MemoryManager service integration",
                    "Designed for staging extracted entities before permanent storage"
                ]
            },
            {
                "name": "Missing Core Services",
                "entityType": "gap_analysis",
                "observations": [
                    "VectorStoreManager - interface between service container and vector implementations",
                    "MemoryManager - session and context management",
                    "KnowledgeGraphManager - Neo4j integration wrapper",
                    "EmbeddingManager - centralized embedding operations",
                    "These manager classes are referenced in code but don't exist"
                ]
            },
            {
                "name": "Missing Specialized Agents",
                "entityType": "gap_analysis",
                "observations": [
                    "ViolationDetectorAgent - Legal violation identification (planned but not built)",
                    "LegalAnalyzerAgent - Constitutional and procedural analysis (planned but not built)",
                    "AutoTaggingAgent - Learning-based document tagging (planned but not built)",
                    "NoteTakingAgent - Context-aware note management (planned but not built)",
                    "Only DocumentProcessorAgent is substantially implemented"
                ]
            },
            {
                "name": "Neo4j Memory Storage",
                "entityType": "memory_system",
                "observations": [
                    "Connection: neo4j://localhost:7687",
                    "Database: CaseDBMS",
                    "User: neo4j with admin privileges",
                    "Available for Claude Code memory persistence",
                    "Can store entities, relationships, and observations from legal analysis",
                    "Supports complex graph queries for legal entity relationships"
                ]
            },
            {
                "name": "Advanced Legal Agents",
                "entityType": "implementation_ready",
                "observations": [
                    "ViolationDetectorAgent: Comprehensive Brady violation, constitutional violation, and prosecutorial misconduct detection",
                    "LegalAnalyzerAgent: IRAC framework analysis, constitutional analysis, precedent analysis",
                    "AutoTaggingAgent: Learning-based document classification with user feedback",
                    "NoteTakingAgent: Context-aware legal note management with citation support",
                    "All agents implement BaseAgent pattern with async processing and error handling"
                ]
            },
            {
                "name": "Manager Classes Implementation",
                "entityType": "implementation_ready",
                "observations": [
                    "VectorStoreManager: Service container interface for vector storage operations",
                    "KnowledgeGraphManager: Neo4j integration wrapper with connection management",
                    "MemoryManager: Session persistence and context management with SQLite",
                    "EmbeddingManager: Centralized embedding operations with model management",
                    "All managers follow service container dependency injection pattern"
                ]
            }
        ],
        "relations": [
            {
                "from": "Legal AI System Architecture",
                "to": "Document Processing System",
                "relationType": "contains"
            },
            {
                "from": "Legal AI System Architecture",
                "to": "Vector Storage System",
                "relationType": "contains"
            },
            {
                "from": "Legal AI System Architecture",
                "to": "Knowledge Graph System",
                "relationType": "contains"
            },
            {
                "from": "Legal AI System Architecture",
                "to": "Memory Management System",
                "relationType": "contains"
            },
            {
                "from": "Missing Core Services",
                "to": "Legal AI System Architecture",
                "relationType": "required_by"
            },
            {
                "from": "Missing Specialized Agents",
                "to": "Legal AI System Architecture",
                "relationType": "required_by"
            },
            {
                "from": "Neo4j Memory Storage",
                "to": "Knowledge Graph System",
                "relationType": "provides_persistence_for"
            },
            {
                "from": "Advanced Legal Agents",
                "to": "Missing Specialized Agents",
                "relationType": "implements"
            },
            {
                "from": "Manager Classes Implementation",
                "to": "Missing Core Services",
                "relationType": "implements"
            },
            {
                "from": "Neo4j Memory Storage",
                "to": "Legal AI System Architecture",
                "relationType": "stores_memories_for"
            }
        ]
    }
    
    # Initialize transfer
    transfer = MemoryTransfer()
    
    try:
        # Connect to Neo4j
        if not transfer.connect():
            logger.error("Failed to connect to Neo4j. Exiting.")
            return
        
        # Clear existing memories
        logger.info("Clearing existing Claude Code memories...")
        transfer.clear_claude_memories()
        
        # Transfer memories
        logger.info("Transferring memories to Neo4j...")
        transfer.transfer_memories(memory_data)
        
        # Verify transfer
        counts = transfer.verify_transfer()
        logger.info(f"Transfer complete! Created {counts['entities']} entities, {counts['observations']} observations, {counts['relationships']} relationships")
        
    except Exception as e:
        logger.error(f"Transfer failed: {e}")
    finally:
        transfer.close()

if __name__ == "__main__":
    main()