"""
Streamlined Entity Extraction Agent - Design.txt Compliant
Uses shared components, DDD principles, structured logging, and error resilience
"""

import asyncio
import logging
import structlog
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from pybreaker import CircuitBreaker

from .base_agent import BaseAgent
from ..core.shared_components import (
    DocumentChunker, measure_performance, performance_tracker, 
    LegalDocumentClassifier, ProcessingCache
)
from ..core.error_recovery import ErrorRecovery
try:
    from ..utils.ontology import (
        LegalEntityType, get_entity_types_for_prompt, 
        validate_entity_attributes, ENTITY_TYPE_MAPPING
    )
except ImportError:
    # Fallback if ontology module not available
    LegalEntityType = None
    get_entity_types_for_prompt = lambda: []
    validate_entity_attributes = lambda x, y: True
    ENTITY_TYPE_MAPPING = {}

# Structured logging
logger = structlog.get_logger()

# Circuit breaker for external services
entity_extraction_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)


# Domain-Driven Design Entities (design.txt recommendation)
@dataclass
class LegalEntity:
    """Core business entity representing a legal entity."""
    entity_id: str
    entity_type: str
    name: str
    attributes: Dict[str, Any]
    confidence_score: float
    source_text: str
    
@dataclass
class EntityExtractionResult:
    """Results from entity extraction with validation metrics and DDD structure."""
    entities: List[LegalEntity]
    validation_summary: Dict[str, int]
    confidence_score: float
    processing_time: float
    model_used: str
    extraction_method: str = "unified_pipeline"
    document_type: Optional[str] = None
    business_context: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [{
                "entity_id": e.entity_id,
                "entity_type": e.entity_type,
                "name": e.name,
                "attributes": e.attributes,
                "confidence_score": e.confidence_score,
                "source_text": e.source_text
            } for e in self.entities],
            "validation_summary": self.validation_summary,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "extraction_method": self.extraction_method,
            "document_type": self.document_type,
            "business_context": self.business_context,
            "extracted_at": datetime.now().isoformat()
        }


class EntityExtractionAgent(BaseAgent):
    """
    Design.txt compliant entity extraction agent with DDD, structured logging, and resilience.
    
    Features:
    - Domain-Driven Design with proper business entities
    - Structured logging with observability metrics
    - Circuit breaker and retry patterns for resilience
    - Separation of concerns between extraction, parsing, and storage
    - Performance optimization with caching and resource management
    """
    
    def __init__(self, services):
        super().__init__(services, "EntityExtraction")
        
        # Shared components (design.txt: separation of concerns)
        self.chunker = DocumentChunker()
        self.classifier = LegalDocumentClassifier()
        self.cache = ProcessingCache()
        self.error_recovery = ErrorRecovery()
        
        # Configuration (following design.txt pattern)
        self.min_confidence = 0.7
        self.max_entities_per_chunk = 50
        self.chunk_overlap = 200
        
        # Entity types from ontology
        if LegalEntityType:
            self.entity_types = [e.value for e in LegalEntityType]
        else:
            self.entity_types = ["Person", "Organization", "Case", "Statute", "Evidence"]
        
        # Structured logging (design.txt requirement)
        self.metrics = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'entities_extracted': 0,
            'average_confidence': 0.0
        }
        
        logger.info("EntityExtraction initialized with design.txt compliance", 
                   agent_type="EntityExtraction",
                   entity_types_count=len(self.entity_types),
                   min_confidence=self.min_confidence)
    
    @measure_performance("entity_extraction")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _process_task(self, task_data, metadata):
        """
        Extract entities with design.txt compliance: DDD, structured logging, error resilience.
        
        Args:
            task_data: Text content or document content
            metadata: Processing options and configuration
        
        Returns:
            EntityExtractionResult with business entities
        """
        start_time = datetime.now()
        document_id = metadata.get("document_id", "unknown")
        
        # Structured logging (design.txt)
        logger.info("entity_extraction_started", 
                   document_id=document_id,
                   timestamp=start_time.isoformat())
        
        try:
            # Extract and validate text content
            text_content = self._extract_text_content(task_data)
            
            if not text_content or len(text_content.strip()) < 10:
                logger.warning("insufficient_text_content", 
                              document_id=document_id,
                              content_length=len(text_content) if text_content else 0)
                return self._create_empty_result("Insufficient text content")
            
            # Document classification (separation of concerns)
            document_type = await self._classify_document(text_content)
            
            # Check cache first (performance optimization)
            cache_key = f"entities_{hash(text_content[:1000])}_{document_type}"
            cached_result = self.cache.get(document_id, cache_key)
            if cached_result:
                logger.info("entity_extraction_cache_hit", 
                           document_id=document_id,
                           cache_key=cache_key)
                return cached_result
            
            # Circuit breaker protection
            entities = await self._extract_with_circuit_breaker(text_content, metadata, document_type)
            
            # Create business result (DDD)
            result = await self._create_business_result(entities, text_content, document_type, start_time)
            
            # Cache the result
            self.cache.set(document_id, cache_key, result)
            
            # Update metrics
            self._update_metrics(result)
            
            logger.info("entity_extraction_completed", 
                       document_id=document_id,
                       entities_count=len(result.entities),
                       confidence_score=result.confidence_score,
                       processing_time=result.processing_time)
            
            return result.to_dict()
            
        except Exception as e:
            # Structured error logging
            logger.error("entity_extraction_failed", 
                        document_id=document_id,
                        error=str(e),
                        error_type=type(e).__name__,
                        processing_time=(datetime.now() - start_time).total_seconds())
            raise
            return self._create_empty_result(f"Processing failed: {str(e)}")
    
    async def _classify_document(self, text_content: str) -> str:
        """Classify document type using shared classifier (separation of concerns)."""
        try:
            return await self.classifier.classify_legal_document(text_content)
        except Exception as e:
            logger.warning("document_classification_failed", error=str(e))
            return "unknown"
    
    @entity_extraction_breaker
    async def _extract_with_circuit_breaker(self, text_content: str, metadata: Dict, document_type: str) -> List[Dict]:
        """Extract entities with circuit breaker protection."""
        return await self._extract_entities_from_chunks(text_content, metadata, document_type)
    
    async def _extract_entities_from_chunks(self, text_content: str, metadata: Dict, document_type: str) -> List[Dict]:
        """Extract entities using chunking strategy for memory efficiency."""
        chunks = self.chunker.chunk_text(
            text_content, 
            chunk_size=3000, 
            overlap=self.chunk_overlap
        )
        
        all_entities = []
        chunk_count = len(chunks)
        
        logger.info("processing_document_chunks", 
                   chunk_count=chunk_count,
                   document_type=document_type)
        
        for i, chunk in enumerate(chunks):
            try:
                # Extract entities from chunk
                chunk_entities = await self._extract_from_single_chunk(
                    chunk, metadata, document_type, i, chunk_count
                )
                all_entities.extend(chunk_entities)
                
                # Rate limiting
                if i < chunk_count - 1:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.warning("chunk_processing_failed", 
                              chunk_index=i,
                              error=str(e))
                continue
        
        # Deduplicate entities
        return self._deduplicate_entities(all_entities)
    
    async def _extract_from_single_chunk(self, chunk: str, metadata: Dict, document_type: str, chunk_index: int, total_chunks: int) -> List[Dict]:
        """Extract entities from a single text chunk."""
        if not hasattr(self.services, 'llm_manager'):
            logger.warning("llm_manager_not_available")
            return []
        
        # Build extraction prompt
        prompt = self._build_extraction_prompt(chunk, document_type)
        
        # Call LLM with error recovery
        response = await self.error_recovery.recover_with_retry(
            self.services.llm_manager.complete,
            prompt,
            context={\"chunk_index\": chunk_index, \"document_type\": document_type}
        )
        
        # Parse and validate entities
        entities = self._parse_llm_response(response, chunk)
        return self._validate_entities(entities)
    
    def _build_extraction_prompt(self, text: str, document_type: str) -> str:
        """Build extraction prompt based on document type and entity types."""
        entity_types_str = \", \".join(self.entity_types)
        
        return f\"\"\"Extract legal entities from this {document_type} document text.
        
Entity types to extract: {entity_types_str}

For each entity, provide:
- entity_type: One of the specified types
- name: The entity name/identifier
- attributes: Relevant attributes as key-value pairs
- confidence: Confidence score (0.0-1.0)
- source_text: The exact text where the entity was found

Text to analyze:
{text}

Return entities in JSON format with the structure above.
\"\"\"
    
    def _parse_llm_response(self, response: str, source_chunk: str) -> List[Dict]:
        """Parse LLM response into entity dictionaries."""
        try:
            import json
            import re
            
            # Try to extract JSON from response
            json_match = re.search(r'\\[.*?\\]', response, re.DOTALL)
            if not json_match:
                return []
            
            entities_data = json.loads(json_match.group())
            
            entities = []
            for entity_data in entities_data:
                if isinstance(entity_data, dict) and 'entity_type' in entity_data:
                    # Add source chunk reference
                    entity_data['source_chunk'] = source_chunk[:200] + '...' if len(source_chunk) > 200 else source_chunk
                    entities.append(entity_data)
            
            return entities
            
        except Exception as e:
            logger.warning("llm_response_parsing_failed", error=str(e))
            return []
    
    def _validate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Validate extracted entities using ontology validation."""
        validated = []
        
        for entity in entities:
            if not entity.get('name') or not entity.get('entity_type'):
                continue
                
            # Check confidence threshold
            confidence = entity.get('confidence', 0.0)
            if confidence < self.min_confidence:
                continue
            
            # Ontology validation if available
            if validate_entity_attributes:
                if not validate_entity_attributes(entity.get('entity_type'), entity.get('attributes', {})):
                    logger.debug("entity_failed_ontology_validation", entity_name=entity.get('name'))
                    continue
            
            validated.append(entity)
        
        return validated[:self.max_entities_per_chunk]  # Limit entities per chunk
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities based on name and type."""
        seen = set()
        deduplicated = []
        
        for entity in entities:
            key = (entity.get('entity_type', ''), entity.get('name', '').lower())
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        return deduplicated
    
    async def _create_business_result(self, entities: List[Dict], text_content: str, document_type: str, start_time: datetime) -> EntityExtractionResult:
        """Create business result using Domain-Driven Design entities."""
        legal_entities = []
        
        for i, entity_data in enumerate(entities):
            legal_entity = LegalEntity(
                entity_id=f\"{entity_data.get('entity_type', 'unknown')}_{i}\",
                entity_type=entity_data.get('entity_type', 'unknown'),
                name=entity_data.get('name', ''),
                attributes=entity_data.get('attributes', {}),
                confidence_score=entity_data.get('confidence', 0.0),
                source_text=entity_data.get('source_text', '')
            )
            legal_entities.append(legal_entity)
        
        # Calculate overall confidence
        if legal_entities:
            avg_confidence = sum(e.confidence_score for e in legal_entities) / len(legal_entities)
        else:
            avg_confidence = 0.0
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return EntityExtractionResult(
            entities=legal_entities,
            validation_summary={
                \"total\": len(entities),
                \"valid\": len(legal_entities),
                \"invalid\": len(entities) - len(legal_entities)
            },
            confidence_score=avg_confidence,
            processing_time=processing_time,
            model_used=getattr(self.services, 'current_model', 'unknown'),
            document_type=document_type,
            business_context=f\"Legal document analysis - {document_type}\"
        )
    
    def _create_empty_result(self, error_msg: str) -> Dict:
        """Create empty result for error cases."""
        return {
            \"success\": False,
            \"error\": error_msg,
            \"entities\": [],
            \"validation_summary\": {\"total\": 0, \"valid\": 0, \"invalid\": 0},
            \"confidence_score\": 0.0,
            \"processing_time\": 0.0
        }
    
    def _update_metrics(self, result: EntityExtractionResult) -> None:
        \"\"\"Update extraction metrics for observability.\"\"\"
        self.metrics['total_extractions'] += 1
        if result.entities:
            self.metrics['successful_extractions'] += 1
            self.metrics['entities_extracted'] += len(result.entities)
            
        # Update running average confidence
        total_extractions = self.metrics['total_extractions']
        current_avg = self.metrics['average_confidence']
        new_confidence = result.confidence_score
        self.metrics['average_confidence'] = ((current_avg * (total_extractions - 1)) + new_confidence) / total_extractions
    
    def _extract_text_content(self, task_data) -> str:
        """Extract text content from various input formats."""
        if isinstance(task_data, str):
            return task_data
        
        if isinstance(task_data, dict):
            # Handle document processor output
            if "content" in task_data and isinstance(task_data["content"], dict):
                return task_data["content"].get("text", "")
            elif "text" in task_data:
                return task_data["text"]
        
        if isinstance(task_data, list):
            # Handle multiple text chunks
            return " ".join(str(chunk) for chunk in task_data)
        
        return str(task_data)
    
    def _convert_to_legacy_format(self, pipeline_result: ProcessingResult, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert unified pipeline results to legacy format."""
        result = {
            "success": True,
            "processing_time": pipeline_result.processing_time,
            "confidence_score": pipeline_result.confidence_score,
            "document_id": pipeline_result.document_id
        }
        
        # Extract entities
        entities = pipeline_result.entities or []
        result["entities"] = entities
        result["entity_count"] = len(entities)
        
        # Validation summary
        if pipeline_result.validation_results:
            result["validation_summary"] = pipeline_result.validation_results
        else:
            # Generate basic validation summary
            valid_entities = [e for e in entities if e.get("confidence", 0) >= self.min_confidence]
            result["validation_summary"] = {
                "total": len(entities),
                "valid": len(valid_entities),
                "invalid": len(entities) - len(valid_entities),
                "confidence_distribution": self._get_confidence_distribution(entities)
            }
        
        # Group entities by type
        result["entities_by_type"] = self._group_entities_by_type(entities)
        
        # Extract metadata from pipeline result
        if pipeline_result.metadata:
            result.update(pipeline_result.metadata)
            
        return result
    
    def _group_entities_by_type(self, entities: List[Dict]) -> Dict[str, List[Dict]]:
        """Group entities by their type for better organization."""
        grouped = {}
        for entity in entities:
            entity_type = entity.get("entity_type", "unknown")
            if entity_type not in grouped:
                grouped[entity_type] = []
            grouped[entity_type].append(entity)
        return grouped
    
    def _get_confidence_distribution(self, entities: List[Dict]) -> Dict[str, int]:
        """Get confidence score distribution for analysis."""
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for entity in entities:
            confidence = entity.get("confidence", 0.0)
            if confidence >= 0.8:
                distribution["high"] += 1
            elif confidence >= 0.6:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
                
        return distribution
    
    # Health check method (design.txt requirement)
    async def health_check(self) -> Dict[str, Any]:
        """Health check for entity extraction service."""
        try:
            # Check LLM availability
            llm_available = hasattr(self.services, 'llm_manager') and self.services.llm_manager is not None
            
            # Check ontology availability
            ontology_available = LegalEntityType is not None
            
            # Check component health
            components_healthy = all([
                self.chunker is not None,
                self.classifier is not None,
                self.cache is not None,
                self.error_recovery is not None
            ])
            
            # Circuit breaker status
            circuit_breaker_status = "closed" if not entity_extraction_breaker.current_state else str(entity_extraction_breaker.current_state)
            
            health_status = {
                "service": "EntityExtractionAgent",
                "status": "healthy" if all([llm_available, components_healthy]) else "degraded",
                "components": {
                    "llm_manager": "available" if llm_available else "unavailable",
                    "ontology": "available" if ontology_available else "unavailable", 
                    "shared_components": "healthy" if components_healthy else "unhealthy",
                    "circuit_breaker": circuit_breaker_status
                },
                "metrics": self.metrics.copy(),
                "configuration": {
                    "min_confidence": self.min_confidence,
                    "max_entities_per_chunk": self.max_entities_per_chunk,
                    "chunk_overlap": self.chunk_overlap,
                    "entity_types_count": len(self.entity_types)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("entity_extraction_health_check", **health_status)
            return health_status
            
        except Exception as e:
            error_status = {
                "service": "EntityExtractionAgent",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            logger.error("entity_extraction_health_check_failed", **error_status)
            return error_status
            result["metadata"] = pipeline_result.metadata
        
        # Create EntityExtractionResult for compatibility
        extraction_result = EntityExtractionResult(
            entities=entities,
            validation_summary=result["validation_summary"],
            confidence_score=pipeline_result.confidence_score,
            processing_time=pipeline_result.processing_time,
            model_used=metadata.get("model_used", "unified_pipeline"),
            extraction_method="unified_pipeline"
        )
        
        result["extraction_result"] = extraction_result.to_dict()
        
        return result
    
    def _group_entities_by_type(self, entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group entities by their type."""
        grouped = {}
        for entity in entities:
            entity_type = entity.get("type", "unknown")
            if entity_type not in grouped:
                grouped[entity_type] = []
            grouped[entity_type].append(entity)
        return grouped
    
    def _get_confidence_distribution(self, entities: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of entity confidence scores."""
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for entity in entities:
            confidence = entity.get("confidence", 0)
            if confidence >= 0.8:
                distribution["high"] += 1
            elif confidence >= 0.6:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution
    
    async def extract_entities_from_text(self, text: str, options: Dict[str, Any] = None) -> EntityExtractionResult:
        """
        Direct entity extraction method for programmatic use.
        
        Args:
            text: Text content to process
            options: Processing options
        
        Returns:
            EntityExtractionResult with extracted entities
        """
        options = options or {}
        result = await self._process_task(text, options)
        
        if result.get("success"):
            return EntityExtractionResult(
                entities=result.get("entities", []),
                validation_summary=result.get("validation_summary", {}),
                confidence_score=result.get("confidence_score", 0.0),
                processing_time=result.get("processing_time", 0.0),
                model_used=options.get("model_used", "unified_pipeline")
            )
        else:
            raise Exception(result.get("error", "Entity extraction failed"))
    
    async def get_extraction_stats(self) -> Dict[str, Any]:
        """Get entity extraction statistics and configuration."""
        return {
            "supported_entity_types": [e.value for e in self.entity_types],
            "min_confidence_threshold": self.min_confidence,
            "configuration": {
                "chunk_size": self.config.get('entity_extraction.chunk_size'),
                "max_entities_per_chunk": self.config.get('entity_extraction.max_entities_per_chunk'),
                "validation_enabled": self.config.get('entity_extraction.validation_enabled')
            },
            "processing_capabilities": {
                "batch_processing": True,
                "async_processing": True,
                "error_recovery": True,
                "ontology_validation": True
            }
        }


# Maintain compatibility exports
__all__ = ["EntityExtractionAgent", "EntityExtractionResult"]