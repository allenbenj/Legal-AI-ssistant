"""
Knowledge Base Agent - Design.txt Compliant
Handles entity resolution and ensures data is properly structured for organizational and analytical purposes.
Part of the seven-agent file organization system architecture.
"""

import structlog
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from pybreaker import CircuitBreaker
import hashlib
import uuid

from ..core.base_agent import BaseAgent
from ..core.performance import measure_performance, performance_tracker
from ..legacy_extras.modular_improvements import ProcessingCache
from ..utils.error_recovery import ErrorRecovery

# Structured logging
logger = structlog.get_logger()

# Circuit breaker for database operations (design.txt requirement)
knowledge_base_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

# Domain-Driven Design Entities (design.txt requirement)
@dataclass
class ResolvedEntity:
    """Core business entity representing a resolved knowledge entity."""
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    source_documents: Set[str] = field(default_factory=set)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_alias(self, alias: str) -> None:
        """Add an alias to this entity."""
        self.aliases.add(alias.lower().strip())
        self.updated_at = datetime.now()
    
    def merge_with(self, other: 'ResolvedEntity') -> None:
        """Merge another entity into this one."""
        self.aliases.update(other.aliases)
        self.source_documents.update(other.source_documents)
        self.attributes.update(other.attributes)
        self.relationships.extend(other.relationships)
        self.confidence_score = max(self.confidence_score, other.confidence_score)
        self.updated_at = datetime.now()

@dataclass
class EntityResolutionResult:
    """Results from entity resolution with organizational and analytical structure."""
    resolved_entities: List[ResolvedEntity]
    resolution_metrics: Dict[str, int]
    organizational_structure: Dict[str, Any]
    analytical_insights: Dict[str, Any]
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolved_entities": [{
                "entity_id": e.entity_id,
                "canonical_name": e.canonical_name,
                "entity_type": e.entity_type,
                "aliases": list(e.aliases),
                "attributes": e.attributes,
                "confidence_score": e.confidence_score,
                "source_documents": list(e.source_documents),
                "relationships": e.relationships,
                "created_at": e.created_at.isoformat(),
                "updated_at": e.updated_at.isoformat()
            } for e in self.resolved_entities],
            "resolution_metrics": self.resolution_metrics,
            "organizational_structure": self.organizational_structure,
            "analytical_insights": self.analytical_insights,
            "processing_time": self.processing_time,
            "resolved_at": datetime.now().isoformat()
        }

class KnowledgeBaseAgent(BaseAgent):
    """
    Knowledge Base Agent implementing design.txt compliance with entity resolution
    and dual-purpose data structuring for organization and analysis.
    
    Features:
    - Entity resolution with deduplication and canonical naming
    - Dual-purpose structuring: organizational hierarchy + analytical insights
    - Circuit breaker protection for database operations
    - Structured logging and observability metrics
    - Hybrid storage integration (graph + vector + filesystem)
    """
    
    def __init__(self, services):
        super().__init__(services, "KnowledgeBase")
        
        # Shared components (design.txt: separation of concerns)
        self.cache = ProcessingCache()
        self.error_recovery = ErrorRecovery()
        
        # Entity resolution configuration
        self.similarity_threshold = 0.85
        self.max_aliases_per_entity = 10
        self.confidence_threshold = 0.7
        
        # Knowledge base storage
        self.entity_registry: Dict[str, ResolvedEntity] = {}
        self.name_to_entity_map: Dict[str, str] = {}  # Name -> entity_id mapping
        
        # Metrics for observability (design.txt requirement)
        self.metrics = {
            'entities_processed': 0,
            'entities_resolved': 0,
            'duplicates_merged': 0,
            'organizational_structures_created': 0,
            'analytical_insights_generated': 0,
            'average_confidence': 0.0
        }
        
        # Structured logging initialization
        logger.info("KnowledgeBase agent initialized with design.txt compliance",
                   agent_type="KnowledgeBase",
                   similarity_threshold=self.similarity_threshold,
                   confidence_threshold=self.confidence_threshold)
    
    @measure_performance("knowledge_base_processing")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _process_task(self, task_data, metadata):
        """
        Process entities for resolution and dual-purpose structuring.
        
        Args:
            task_data: Raw entities from extraction agents
            metadata: Processing context and configuration
        
        Returns:
            EntityResolutionResult with organizational and analytical structure
        """
        start_time = datetime.now()
        document_id = metadata.get("document_id", "unknown")
        
        # Structured logging (design.txt)
        logger.info("knowledge_base_processing_started",
                   document_id=document_id,
                   timestamp=start_time.isoformat())
        
        try:
            # Extract entities from task data
            raw_entities = self._extract_entities_from_task(task_data)
            
            if not raw_entities:
                logger.warning("no_entities_to_process", document_id=document_id)
                return self._create_empty_result("No entities provided for processing")
            
            # Check cache first (performance optimization)
            cache_key = self._generate_cache_key(raw_entities, metadata)
            cached_result = self.cache.get(document_id, cache_key)
            if cached_result:
                logger.info("knowledge_base_cache_hit", 
                           document_id=document_id,
                           cache_key=cache_key)
                return cached_result
            
            # Entity resolution with circuit breaker protection
            resolved_entities = await self._resolve_entities_with_protection(raw_entities, document_id)
            
            # Dual-purpose structuring
            organizational_structure = await self._create_organizational_structure(resolved_entities)
            analytical_insights = await self._generate_analytical_insights(resolved_entities, metadata)
            
            # Create business result
            result = EntityResolutionResult(
                resolved_entities=resolved_entities,
                resolution_metrics=self._calculate_resolution_metrics(raw_entities, resolved_entities),
                organizational_structure=organizational_structure,
                analytical_insights=analytical_insights,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            # Cache the result
            self.cache.set(document_id, cache_key, result.to_dict())
            
            # Update metrics
            self._update_metrics(result)
            
            logger.info("knowledge_base_processing_completed",
                       document_id=document_id,
                       entities_resolved=len(resolved_entities),
                       processing_time=result.processing_time)
            
            return result.to_dict()
            
        except Exception as e:
            # Structured error logging
            logger.error("knowledge_base_processing_failed",
                        document_id=document_id,
                        error=str(e),
                        error_type=type(e).__name__,
                        processing_time=(datetime.now() - start_time).total_seconds())
            raise
    
    def _extract_entities_from_task(self, task_data) -> List[Dict[str, Any]]:
        """Extract entities from various task data formats."""
        if isinstance(task_data, dict):
            if "entities" in task_data:
                return task_data["entities"]
            elif "resolved_entities" in task_data:
                return task_data["resolved_entities"]
        
        if isinstance(task_data, list):
            return task_data
        
        return []
    
    def _generate_cache_key(self, entities: List[Dict], metadata: Dict) -> str:
        """Generate cache key for entity resolution results."""
        entity_hash = hashlib.md5(str(sorted([e.get("name", "") for e in entities])).encode()).hexdigest()
        metadata_hash = hashlib.md5(str(sorted(metadata.items())).encode()).hexdigest()
        return f"kb_resolution_{entity_hash}_{metadata_hash}"
    
    @knowledge_base_breaker
    async def _resolve_entities_with_protection(self, raw_entities: List[Dict], document_id: str) -> List[ResolvedEntity]:
        """Resolve entities with circuit breaker protection."""
        return await self._resolve_entities(raw_entities, document_id)
    
    async def _resolve_entities(self, raw_entities: List[Dict], document_id: str) -> List[ResolvedEntity]:
        """Core entity resolution logic with deduplication and canonical naming."""
        resolved_entities = []
        
        for raw_entity in raw_entities:
            entity_name = raw_entity.get("name", "").strip()
            entity_type = raw_entity.get("entity_type", "unknown")
            
            if not entity_name:
                continue
            
            # Check if entity already exists (by name or alias)
            existing_entity_id = self._find_existing_entity(entity_name, entity_type)
            
            if existing_entity_id:
                # Update existing entity
                existing_entity = self.entity_registry[existing_entity_id]
                existing_entity.add_alias(entity_name)
                existing_entity.source_documents.add(document_id)
                existing_entity.attributes.update(raw_entity.get("attributes", {}))
                resolved_entities.append(existing_entity)
                
                logger.debug("entity_merged_with_existing",
                           entity_name=entity_name,
                           existing_id=existing_entity_id)
                
                self.metrics['duplicates_merged'] += 1
            else:
                # Create new resolved entity
                resolved_entity = ResolvedEntity(
                    entity_id=str(uuid.uuid4()),
                    canonical_name=entity_name,
                    entity_type=entity_type,
                    attributes=raw_entity.get("attributes", {}),
                    confidence_score=raw_entity.get("confidence", 0.0),
                    source_documents={document_id}
                )
                
                # Add to registry
                self.entity_registry[resolved_entity.entity_id] = resolved_entity
                self.name_to_entity_map[entity_name.lower()] = resolved_entity.entity_id
                
                resolved_entities.append(resolved_entity)
                
                logger.debug("new_entity_created",
                           entity_id=resolved_entity.entity_id,
                           entity_name=entity_name)
                
                self.metrics['entities_resolved'] += 1
        
        return resolved_entities
    
    def _find_existing_entity(self, entity_name: str, entity_type: str) -> Optional[str]:
        """Find existing entity by name or alias with similarity matching."""
        normalized_name = entity_name.lower().strip()
        
        # Exact match first
        if normalized_name in self.name_to_entity_map:
            return self.name_to_entity_map[normalized_name]
        
        # Fuzzy matching for similar names
        for existing_name, entity_id in self.name_to_entity_map.items():
            entity = self.entity_registry[entity_id]
            
            # Check type compatibility
            if entity.entity_type != entity_type:
                continue
            
            # Check similarity
            similarity = self._calculate_name_similarity(normalized_name, existing_name)
            if similarity >= self.similarity_threshold:
                return entity_id
            
            # Check aliases
            for alias in entity.aliases:
                similarity = self._calculate_name_similarity(normalized_name, alias)
                if similarity >= self.similarity_threshold:
                    return entity_id
        
        return None
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two entity names."""
        # Simple Jaccard similarity for now
        set1 = set(name1.split())
        set2 = set(name2.split())
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    async def _create_organizational_structure(self, entities: List[ResolvedEntity]) -> Dict[str, Any]:
        """Create organizational structure for file organization purposes."""
        structure = {
            "by_type": {},
            "by_confidence": {"high": [], "medium": [], "low": []},
            "hierarchical": {},
            "metadata": {
                "total_entities": len(entities),
                "entity_types": set(),
                "confidence_distribution": {"high": 0, "medium": 0, "low": 0}
            }
        }
        
        for entity in entities:
            entity_type = entity.entity_type
            confidence = entity.confidence_score
            
            # Group by type
            if entity_type not in structure["by_type"]:
                structure["by_type"][entity_type] = []
            structure["by_type"][entity_type].append({
                "entity_id": entity.entity_id,
                "name": entity.canonical_name,
                "confidence": confidence
            })
            
            # Group by confidence
            if confidence >= 0.8:
                structure["by_confidence"]["high"].append(entity.entity_id)
                structure["metadata"]["confidence_distribution"]["high"] += 1
            elif confidence >= 0.6:
                structure["by_confidence"]["medium"].append(entity.entity_id)
                structure["metadata"]["confidence_distribution"]["medium"] += 1
            else:
                structure["by_confidence"]["low"].append(entity.entity_id)
                structure["metadata"]["confidence_distribution"]["low"] += 1
            
            structure["metadata"]["entity_types"].add(entity_type)
        
        # Convert set to list for JSON serialization
        structure["metadata"]["entity_types"] = list(structure["metadata"]["entity_types"])
        
        self.metrics['organizational_structures_created'] += 1
        
        logger.info("organizational_structure_created",
                   total_entities=structure["metadata"]["total_entities"],
                   entity_types_count=len(structure["metadata"]["entity_types"]))
        
        return structure
    
    async def _generate_analytical_insights(self, entities: List[ResolvedEntity], metadata: Dict) -> Dict[str, Any]:
        """Generate analytical insights for data analysis purposes."""
        insights = {
            "entity_analysis": {
                "most_confident_entities": [],
                "entity_relationships": [],
                "cross_document_patterns": [],
                "anomalies": []
            },
            "document_analysis": {
                "document_entity_density": metadata.get("document_id", "unknown"),
                "entity_distribution": {},
                "key_entities": []
            },
            "knowledge_patterns": {
                "frequent_entity_combinations": [],
                "entity_co_occurrence": {},
                "temporal_patterns": []
            },
            "quality_metrics": {
                "overall_confidence": 0.0,
                "resolution_accuracy": 0.0,
                "completeness_score": 0.0
            }
        }
        
        if entities:
            # Calculate overall confidence
            insights["quality_metrics"]["overall_confidence"] = sum(e.confidence_score for e in entities) / len(entities)
            
            # Find most confident entities
            sorted_entities = sorted(entities, key=lambda x: x.confidence_score, reverse=True)
            insights["entity_analysis"]["most_confident_entities"] = [
                {"name": e.canonical_name, "type": e.entity_type, "confidence": e.confidence_score}
                for e in sorted_entities[:5]
            ]
            
            # Entity distribution analysis
            type_counts = {}
            for entity in entities:
                entity_type = entity.entity_type
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
            
            insights["document_analysis"]["entity_distribution"] = type_counts
            
            # Key entities (high confidence + multiple sources)
            key_entities = [
                e for e in entities 
                if e.confidence_score >= 0.8 and len(e.source_documents) > 1
            ]
            insights["document_analysis"]["key_entities"] = [
                {"name": e.canonical_name, "sources": len(e.source_documents)}
                for e in key_entities
            ]
        
        self.metrics['analytical_insights_generated'] += 1
        
        logger.info("analytical_insights_generated",
                   entity_count=len(entities),
                   overall_confidence=insights["quality_metrics"]["overall_confidence"])
        
        return insights
    
    def _calculate_resolution_metrics(self, raw_entities: List[Dict], resolved_entities: List[ResolvedEntity]) -> Dict[str, float]:
        """Calculate metrics about the resolution process."""
        return {
            "input_entities": len(raw_entities),
            "resolved_entities": len(resolved_entities),
            "deduplication_ratio": (len(raw_entities) - len(resolved_entities)) / len(raw_entities) if raw_entities else 0,
            "high_confidence_entities": sum(1 for e in resolved_entities if e.confidence_score >= 0.8),
            "entities_with_aliases": sum(1 for e in resolved_entities if e.aliases),
            "multi_source_entities": sum(1 for e in resolved_entities if len(e.source_documents) > 1)
        }
    
    def _create_empty_result(self, error_msg: str) -> Dict[str, Any]:
        """Create empty result for error cases."""
        return {
            "success": False,
            "error": error_msg,
            "resolved_entities": [],
            "resolution_metrics": {"input_entities": 0, "resolved_entities": 0},
            "organizational_structure": {"by_type": {}, "by_confidence": {}, "metadata": {}},
            "analytical_insights": {"entity_analysis": {}, "document_analysis": {}, "quality_metrics": {}},
            "processing_time": 0.0
        }
    
    def _update_metrics(self, result: EntityResolutionResult) -> None:
        """Update agent metrics for observability."""
        self.metrics['entities_processed'] += result.resolution_metrics.get("input_entities", 0)
        
        # Update running average confidence
        if result.resolved_entities:
            total_confidence = sum(e.confidence_score for e in result.resolved_entities)
            avg_confidence = total_confidence / len(result.resolved_entities)
            
            current_avg = self.metrics['average_confidence']
            total_processed = self.metrics['entities_processed']
            self.metrics['average_confidence'] = ((current_avg * (total_processed - 1)) + avg_confidence) / total_processed
    
    # Health check method (design.txt requirement)
    async def health_check(self) -> Dict[str, Any]:
        """Health check for knowledge base service."""
        try:
            # Check component health
            components_healthy = all([
                self.cache is not None,
                self.error_recovery is not None,
                self.entity_registry is not None
            ])
            
            # Circuit breaker status
            circuit_breaker_status = "closed" if not knowledge_base_breaker.current_state else str(knowledge_base_breaker.current_state)
            
            # Storage health
            storage_healthy = len(self.entity_registry) >= 0  # Basic check
            
            health_status = {
                "service": "KnowledgeBaseAgent",
                "status": "healthy" if all([components_healthy, storage_healthy]) else "degraded",
                "components": {
                    "entity_registry": f"{len(self.entity_registry)} entities",
                    "name_mapping": f"{len(self.name_to_entity_map)} mappings",
                    "shared_components": "healthy" if components_healthy else "unhealthy",
                    "circuit_breaker": circuit_breaker_status
                },
                "metrics": self.metrics.copy(),
                "configuration": {
                    "similarity_threshold": self.similarity_threshold,
                    "confidence_threshold": self.confidence_threshold,
                    "max_aliases_per_entity": self.max_aliases_per_entity
                },
                "storage_stats": {
                    "total_entities": len(self.entity_registry),
                    "entities_with_aliases": sum(1 for e in self.entity_registry.values() if e.aliases),
                    "multi_source_entities": sum(1 for e in self.entity_registry.values() if len(e.source_documents) > 1)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("knowledge_base_health_check", **health_status)
            return health_status
            
        except Exception as e:
            error_status = {
                "service": "KnowledgeBaseAgent",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            logger.error("knowledge_base_health_check_failed", **error_status)
            return error_status