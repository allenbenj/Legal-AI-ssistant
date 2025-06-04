"""
Ontology-driven legal entity and relationship extraction agent.

This agent performs sophisticated legal domain knowledge extraction using
a comprehensive legal ontology to identify entities, relationships, and
structured legal information from processed documents.
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path

from ..core.base_agent import BaseAgent
from ..core.types import ProcessingResult, LegalDocument
from ..utils.ontology import (
    LegalEntityType, LegalRelationshipType, 
    get_entity_types_for_prompt, get_relationship_types_for_prompt,
    get_extraction_prompt
)


@dataclass
class ExtractedEntity:
    """Represents an extracted legal entity with ontology classification."""
    entity_type: str
    entity_id: str
    attributes: Dict[str, Any]
    confidence: float
    source_text: str
    span: Tuple[int, int]  # Character positions in text
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedRelationship:
    """Represents an extracted relationship between legal entities."""
    relationship_type: str
    source_entity: str
    target_entity: str
    properties: Dict[str, Any]
    confidence: float
    source_text: str
    span: Tuple[int, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OntologyExtractionResult:
    """Complete result of ontology-driven extraction."""
    document_id: str
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]
    extraction_metadata: Dict[str, Any]
    processing_time: float
    confidence_scores: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_id': self.document_id,
            'entities': [e.to_dict() for e in self.entities],
            'relationships': [r.to_dict() for r in self.relationships],
            'extraction_metadata': self.extraction_metadata,
            'processing_time': self.processing_time,
            'confidence_scores': self.confidence_scores
        }


class OntologyExtractionAgent(BaseAgent):
    """
    Agent for ontology-driven legal entity and relationship extraction.
    
    This agent analyzes processed legal documents and extracts structured
    information based on a comprehensive legal ontology, identifying:
    - Legal entities (persons, cases, documents, evidence, etc.)
    - Legal relationships (filed_by, ruled_by, supports, refutes, etc.)
    - Temporal and jurisdictional information
    - Evidence chains and procedural connections
    """
    
    def __init__(self, services, **config):
        super().__init__(services, **config)
        self.name = "OntologyExtractionAgent"
        self.description = "Performs ontology-driven legal entity and relationship extraction"
        
        # Entity extraction patterns
        self.entity_patterns = self._build_entity_patterns()
        self.relationship_patterns = self._build_relationship_patterns()
        
        # Extraction configuration
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.max_entities_per_type = config.get('max_entities_per_type', 50)
        self.enable_coreference = config.get('enable_coreference', True)
        
    def _build_entity_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build regex patterns for entity extraction based on legal ontology."""
        patterns = {}
        
        # Person patterns
        patterns['PERSON'] = [
            {
                'pattern': r'\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                'attributes': ['name'],
                'context_keywords': ['defendant', 'plaintiff', 'witness', 'attorney', 'judge']
            }
        ]
        
        # Case patterns
        patterns['CASE'] = [
            {
                'pattern': r'(?:Case|Matter|Docket)\s+(?:No\.?\s*)?([A-Z0-9-]+)',
                'attributes': ['title'],
                'context_keywords': ['case', 'matter', 'docket', 'proceeding']
            },
            {
                'pattern': r'([A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+)',
                'attributes': ['title'],
                'context_keywords': ['versus', 'against']
            }
        ]
        
        # Legal Document patterns
        patterns['LEGALDOCUMENT'] = [
            {
                'pattern': r'\b(Motion|Order|Pleading|Brief|Affidavit|Complaint|Answer|Counterclaim)\b',
                'attributes': ['title'],
                'context_keywords': ['filed', 'submitted', 'dated']
            }
        ]
        
        # Evidence Item patterns
        patterns['EVIDENCEITEM'] = [
            {
                'pattern': r'\b(?:Exhibit|Evidence|Item)\s+([A-Z0-9-]+)\b',
                'attributes': ['description'],
                'context_keywords': ['evidence', 'exhibit', 'proof', 'submitted']
            }
        ]
        
        # Date patterns for temporal entities
        patterns['DATE'] = [
            {
                'pattern': r'\b(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
                'attributes': ['date'],
                'context_keywords': ['on', 'dated', 'filed', 'occurred']
            }
        ]
        
        # Statement patterns
        patterns['STATEMENT'] = [
            {
                'pattern': r'(?:testified|stated|declared|affirmed)\s+(?:that\s+)?(.{10,200})',
                'attributes': ['verbatim'],
                'context_keywords': ['testimony', 'statement', 'declaration']
            }
        ]
        
        # Court patterns
        patterns['COURT'] = [
            {
                'pattern': r'\b([A-Z][a-z]+\s+(?:District|Superior|Circuit|Municipal|County|Federal)\s+Court)\b',
                'attributes': ['name'],
                'context_keywords': ['court', 'jurisdiction', 'presided']
            }
        ]
        
        # Judge patterns
        patterns['JUDGE'] = [
            {
                'pattern': r'(?:Judge|Justice|Hon\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                'attributes': ['name'],
                'context_keywords': ['presiding', 'ruled', 'ordered']
            }
        ]
        
        return patterns
    
    def _build_relationship_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build patterns for relationship extraction."""
        patterns = {}
        
        patterns['FILED_BY'] = [
            {
                'pattern': r'(.+?)\s+filed\s+(.+)',
                'source_group': 1,
                'target_group': 2,
                'properties': ['filed_date']
            }
        ]
        
        patterns['RULED_BY'] = [
            {
                'pattern': r'(.+?)\s+ruled\s+(?:on\s+)?(.+)',
                'source_group': 1,
                'target_group': 2,
                'properties': ['ruled_date']
            }
        ]
        
        patterns['SUPPORTS'] = [
            {
                'pattern': r'(.+?)\s+(?:supports|corroborates|confirms)\s+(.+)',
                'source_group': 1,
                'target_group': 2,
                'properties': ['confidence', 'analysis_method']
            }
        ]
        
        patterns['REFUTES'] = [
            {
                'pattern': r'(.+?)\s+(?:refutes|contradicts|disproves)\s+(.+)',
                'source_group': 1,
                'target_group': 2,
                'properties': ['confidence', 'analysis_method']
            }
        ]
        
        patterns['PRESIDED_BY'] = [
            {
                'pattern': r'(?:Judge|Justice)\s+(.+?)\s+presided\s+over\s+(.+)',
                'source_group': 1,
                'target_group': 2,
                'properties': ['session_date']
            }
        ]
        
        return patterns
    
    async def process_document(self, document: LegalDocument) -> OntologyExtractionResult:
        """
        Extract legal entities and relationships from a processed document.
        
        Args:
            document: Processed legal document with extracted text
            
        Returns:
            OntologyExtractionResult with extracted entities and relationships
        """
        start_time = datetime.now()
        
        try:
            # Extract text content
            text_content = self._extract_text_content(document)
            if not text_content:
                return self._create_empty_result(document.id, start_time)
            
            # Perform entity extraction
            entities = await self._extract_entities(text_content)
            
            # Perform relationship extraction
            relationships = await self._extract_relationships(text_content, entities)
            
            # Post-process and validate results
            entities = self._post_process_entities(entities)
            relationships = self._validate_relationships(relationships, entities)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(entities, relationships)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return OntologyExtractionResult(
                document_id=document.id,
                entities=entities,
                relationships=relationships,
                extraction_metadata={
                    'text_length': len(text_content),
                    'extraction_method': 'pattern_based_with_llm_validation',
                    'ontology_version': '1.0',
                    'agent_version': self.version
                },
                processing_time=processing_time,
                confidence_scores=confidence_scores
            )
            
        except Exception as e:
            self.logger.error(f"Error in ontology extraction for {document.id}: {e}")
            return self._create_empty_result(document.id, start_time)
    
    def _extract_text_content(self, document: LegalDocument) -> str:
        """Extract text content from document."""
        if hasattr(document, 'processed_content') and document.processed_content:
            return document.processed_content.get('text', '')
        elif hasattr(document, 'content'):
            return document.content
        else:
            return ''
    
    async def _extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract legal entities using pattern matching and LLM validation."""
        entities = []
        entity_id_counter = 0
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern_config in patterns:
                pattern = pattern_config['pattern']
                attributes = pattern_config['attributes']
                context_keywords = pattern_config.get('context_keywords', [])
                
                # Find pattern matches
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Check context relevance
                    context_score = self._calculate_context_relevance(
                        text, match.span(), context_keywords
                    )
                    
                    if context_score < 0.3:  # Skip low-relevance matches
                        continue
                    
                    # Extract entity attributes
                    entity_attrs = {}
                    if attributes and len(attributes) > 0:
                        entity_attrs[attributes[0]] = match.group(1) if match.groups() else match.group(0)
                    
                    # Create entity
                    entity = ExtractedEntity(
                        entity_type=entity_type,
                        entity_id=f"{entity_type}_{entity_id_counter}",
                        attributes=entity_attrs,
                        confidence=context_score,
                        source_text=match.group(0),
                        span=match.span()
                    )
                    
                    entities.append(entity)
                    entity_id_counter += 1
        
        # Use LLM for validation and enhancement
        if self.services.llm_provider:
            entities = await self._validate_entities_with_llm(text, entities)
        
        return entities
    
    async def _extract_relationships(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Extract relationships between entities using patterns and LLM assistance."""
        relationships = []
        relationship_id_counter = 0
        
        # First, extract using pattern matching
        relationships.extend(self._extract_relationships_by_patterns(text, entities))
        
        # Then enhance with LLM extraction for complex relationships
        if self.services.llm_provider and entities:
            llm_relationships = await self._extract_relationships_with_llm(text, entities)
            relationships.extend(llm_relationships)
        
        # Remove duplicates and validate
        relationships = self._deduplicate_relationships(relationships)
        
        return relationships
    
    def _extract_relationships_by_patterns(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Extract relationships using predefined patterns."""
        relationships = []
        
        for rel_type, patterns in self.relationship_patterns.items():
            for pattern_config in patterns:
                pattern = pattern_config['pattern']
                source_group = pattern_config.get('source_group', 1)
                target_group = pattern_config.get('target_group', 2)
                properties = pattern_config.get('properties', [])
                
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    if len(match.groups()) >= max(source_group, target_group):
                        source_text = match.group(source_group).strip()
                        target_text = match.group(target_group).strip()
                        
                        # Try to map to extracted entities
                        source_entity = self._find_entity_by_text(source_text, entities)
                        target_entity = self._find_entity_by_text(target_text, entities)
                        
                        if source_entity and target_entity:
                            relationship = ExtractedRelationship(
                                relationship_type=rel_type,
                                source_entity=source_entity.entity_id,
                                target_entity=target_entity.entity_id,
                                properties={},
                                confidence=0.8,  # Base confidence for pattern matches
                                source_text=match.group(0),
                                span=match.span()
                            )
                            
                            relationships.append(relationship)
        
        return relationships
    
    async def _extract_relationships_with_llm(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Use LLM to extract complex relationships between entities."""
        if len(entities) < 2:
            return []
        
        # Prepare entity list for LLM
        entity_list = "\n".join([
            f"- {e.entity_id}: {e.entity_type} - '{e.source_text}'"
            for e in entities[:20]  # Limit for prompt size
        ])
        
        relationship_guidance = get_relationship_types_for_prompt()
        
        prompt = f"""
        RELATIONSHIP EXTRACTION TASK:
        
        Given the following legal entities and text, identify relationships between them using the legal ontology:
        
        RELATIONSHIP TYPES AVAILABLE:
        {relationship_guidance}
        
        EXTRACTED ENTITIES:
        {entity_list}
        
        TEXT TO ANALYZE:
        {text[:2000]}...
        
        Instructions:
        1. Identify relationships between the listed entities based on the text
        2. Use only the relationship types from the ontology above
        3. Look for the specific phrases mentioned in the prompt hints
        4. Include any relevant properties/dates mentioned
        5. Assign confidence scores based on text clarity
        
        Return a JSON array of relationships in this format:
        [
          {{
            "relationship_type": "FILED_BY",
            "source_entity_id": "PERSON_1",
            "target_entity_id": "LEGALDOCUMENT_1", 
            "properties": {{"filed_date": "2024-01-15"}},
            "confidence": 0.9,
            "source_text": "Motion filed by John Smith on January 15, 2024"
          }}
        ]
        
        Only include relationships with confidence > 0.7.
        """
        
        try:
            response = await self.services.llm_provider.generate_response(
                prompt=prompt,
                model_params={'temperature': 0.1, 'max_tokens': 2500}
            )
            
            return self._parse_llm_relationships_response(response, entities)
            
        except Exception as e:
            self.logger.warning(f"LLM relationship extraction failed: {e}")
            return []
    
    def _parse_llm_relationships_response(self, response: str, entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Parse LLM relationship extraction response."""
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return []
            
            relationships_data = json.loads(json_match.group())
            relationships = []
            
            # Create entity ID lookup
            entity_id_lookup = {e.entity_id: e for e in entities}
            
            for item in relationships_data:
                if item.get('confidence', 0) > 0.7:
                    source_id = item.get('source_entity_id')
                    target_id = item.get('target_entity_id')
                    
                    # Validate entity IDs exist
                    if source_id in entity_id_lookup and target_id in entity_id_lookup:
                        relationship = ExtractedRelationship(
                            relationship_type=item['relationship_type'],
                            source_entity=source_id,
                            target_entity=target_id,
                            properties=item.get('properties', {}),
                            confidence=item['confidence'],
                            source_text=item.get('source_text', ''),
                            span=(0, 0)  # Would need to recalculate spans
                        )
                        relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM relationships response: {e}")
            return []
    
    def _deduplicate_relationships(self, relationships: List[ExtractedRelationship]) -> List[ExtractedRelationship]:
        """Remove duplicate relationships."""
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            # Create a unique key for the relationship
            key = (rel.relationship_type, rel.source_entity, rel.target_entity)
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        return unique_relationships
    
    def _find_entity_by_text(self, text: str, entities: List[ExtractedEntity]) -> Optional[ExtractedEntity]:
        """Find entity by matching text content."""
        text_lower = text.lower()
        
        # Exact match first
        for entity in entities:
            if entity.source_text.lower() == text_lower:
                return entity
        
        # Partial match
        for entity in entities:
            if text_lower in entity.source_text.lower() or entity.source_text.lower() in text_lower:
                return entity
        
        return None
    
    def _calculate_context_relevance(self, text: str, span: Tuple[int, int], keywords: List[str]) -> float:
        """Calculate relevance score based on surrounding context."""
        if not keywords:
            return 0.5  # Default score when no context keywords
        
        # Extract context window around the match
        start, end = span
        context_window = 100  # Characters before and after
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        context = text[context_start:context_end].lower()
        
        # Count keyword matches in context
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in context)
        
        return min(1.0, keyword_matches / len(keywords) + 0.3)
    
    async def _validate_entities_with_llm(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Use LLM to validate and enhance extracted entities with ontology guidance."""
        if not entities or not self.services.llm_provider:
            return entities
        
        # Prepare validation prompt with ontology guidance
        entity_summary = "\n".join([
            f"- {e.entity_type}: '{e.source_text}' (confidence: {e.confidence:.2f})"
            for e in entities[:15]  # Limit for prompt size
        ])
        
        extraction_guidance = get_extraction_prompt()
        
        prompt = f"""
        {extraction_guidance}
        
        VALIDATION TASK:
        Review the following legal entities extracted from a document. For each entity:
        1. Verify correct classification according to the legal ontology above
        2. Check if extracted attributes match the entity type requirements
        3. Assess confidence based on context and clarity
        4. Suggest corrections if needed
        
        TEXT EXCERPT (first 1500 chars):
        {text[:1500]}...
        
        EXTRACTED ENTITIES TO VALIDATE:
        {entity_summary}
        
        Return a JSON array of validated entities in this format:
        [
          {{
            "entity_type": "PERSON",
            "source_text": "Judge Smith", 
            "confidence": 0.95,
            "attributes": {{"name": "Judge Smith"}},
            "validation_notes": "Correctly identified as judicial figure"
          }}
        ]
        
        Only include entities with confidence > 0.6.
        """
        
        try:
            response = await self.services.llm_provider.generate_response(
                prompt=prompt,
                model_params={'temperature': 0.1, 'max_tokens': 3000}
            )
            
            # Parse LLM response and update entities
            validated_entities = self._parse_llm_validation_response(response, entities)
            return validated_entities if validated_entities else entities
            
        except Exception as e:
            self.logger.warning(f"LLM validation failed: {e}")
            # Return original entities with slight confidence adjustment
            for entity in entities:
                entity.confidence = min(1.0, entity.confidence + 0.05)
            return entities
    
    def _parse_llm_validation_response(self, response: str, original_entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Parse LLM validation response and update entities."""
        try:
            import json
            import re
            
            # Extract JSON from response (handle cases where LLM adds extra text)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return original_entities
            
            validated_data = json.loads(json_match.group())
            updated_entities = []
            
            for item in validated_data:
                if item.get('confidence', 0) > 0.6:
                    entity = ExtractedEntity(
                        entity_type=item['entity_type'],
                        entity_id=f"{item['entity_type']}_{len(updated_entities)}",
                        attributes=item.get('attributes', {}),
                        confidence=item['confidence'],
                        source_text=item['source_text'],
                        span=(0, 0)  # Would need to recalculate spans
                    )
                    updated_entities.append(entity)
            
            return updated_entities if updated_entities else original_entities
            
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM validation response: {e}")
            return original_entities
    
    def _post_process_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Post-process entities to remove duplicates and improve quality."""
        # Remove low-confidence entities
        entities = [e for e in entities if e.confidence >= self.confidence_threshold]
        
        # Remove duplicates based on text similarity
        unique_entities = []
        seen_texts = set()
        
        for entity in sorted(entities, key=lambda x: x.confidence, reverse=True):
            text_key = entity.source_text.lower().strip()
            if text_key not in seen_texts:
                unique_entities.append(entity)
                seen_texts.add(text_key)
        
        # Limit entities per type
        type_counts = {}
        filtered_entities = []
        
        for entity in unique_entities:
            count = type_counts.get(entity.entity_type, 0)
            if count < self.max_entities_per_type:
                filtered_entities.append(entity)
                type_counts[entity.entity_type] = count + 1
        
        return filtered_entities
    
    def _validate_relationships(self, relationships: List[ExtractedRelationship], 
                              entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Validate that relationships reference valid entities."""
        entity_ids = {e.entity_id for e in entities}
        
        valid_relationships = []
        for rel in relationships:
            if rel.source_entity in entity_ids and rel.target_entity in entity_ids:
                valid_relationships.append(rel)
        
        return valid_relationships
    
    def _calculate_confidence_scores(self, entities: List[ExtractedEntity], 
                                   relationships: List[ExtractedRelationship]) -> Dict[str, float]:
        """Calculate overall confidence scores for the extraction."""
        if not entities:
            return {'overall': 0.0, 'entities': 0.0, 'relationships': 0.0}
        
        entity_confidence = sum(e.confidence for e in entities) / len(entities)
        
        if relationships:
            relationship_confidence = sum(r.confidence for r in relationships) / len(relationships)
        else:
            relationship_confidence = 0.0
        
        overall_confidence = (entity_confidence + relationship_confidence) / 2
        
        return {
            'overall': overall_confidence,
            'entities': entity_confidence,
            'relationships': relationship_confidence,
            'entity_count': len(entities),
            'relationship_count': len(relationships)
        }
    
    def _create_empty_result(self, document_id: str, start_time: datetime) -> OntologyExtractionResult:
        """Create empty result for failed extractions."""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return OntologyExtractionResult(
            document_id=document_id,
            entities=[],
            relationships=[],
            extraction_metadata={
                'extraction_method': 'failed',
                'error': 'No extractable content found'
            },
            processing_time=processing_time,
            confidence_scores={'overall': 0.0, 'entities': 0.0, 'relationships': 0.0}
        )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status."""
        base_status = await super().get_health_status()
        
        base_status.update({
            'ontology_patterns': {
                'entity_types': len(self.entity_patterns),
                'relationship_types': len(self.relationship_patterns)
            },
            'configuration': {
                'confidence_threshold': self.confidence_threshold,
                'max_entities_per_type': self.max_entities_per_type,
                'enable_coreference': self.enable_coreference
            }
        })
        
        return base_status