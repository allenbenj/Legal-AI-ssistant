"""
SemanticAnalysisAgent - Document summarization and legal topic identification.

Provides comprehensive semantic analysis of legal documents including summarization,
key legal topic identification, concept extraction, and contextual understanding
using legal ontology guidance.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .base_agent import BaseAgent
from ..utils.ontology import LegalEntityType, get_entity_types_for_prompt
from ..core.llm_providers import LLMProviderManager
from ..core.model_switcher import ModelSwitcher, TaskComplexity


@dataclass
class SemanticAnalysisResult:
    """Results from semantic analysis of legal document."""
    document_summary: str
    key_topics: List[Dict[str, Any]]
    legal_concepts: List[Dict[str, Any]]
    content_classification: Dict[str, Any]
    semantic_metadata: Dict[str, Any]
    confidence_score: float
    processing_time: float
    model_used: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_summary": self.document_summary,
            "key_topics": self.key_topics,
            "legal_concepts": self.legal_concepts,
            "content_classification": self.content_classification,
            "semantic_metadata": self.semantic_metadata,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "analyzed_at": datetime.now().isoformat()
        }


class SemanticAnalysisAgent(BaseAgent):
    """
    Comprehensive semantic analysis agent for legal documents.
    
    Features:
    - Document summarization with legal focus
    - Key legal topic identification and classification
    - Legal concept extraction using ontology
    - Content type classification and metadata extraction
    - Context-aware analysis with entity integration
    - Multi-complexity analysis with appropriate model selection
    """
    
    def __init__(self, llm_manager: LLMProviderManager, model_switcher: ModelSwitcher):
        super().__init__("SemanticAnalysisAgent")
        self.llm_manager = llm_manager
        self.model_switcher = model_switcher
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.min_confidence_threshold = 0.7
        self.max_topics_per_analysis = 15
        self.summary_max_length = 1000
        self.include_legal_concepts = True
        self.classify_content_type = True
        
        # Semantic analysis prompt
        self.semantic_prompt_template = """Perform comprehensive semantic analysis of the legal document using legal ontology concepts.

LEGAL CONCEPT SCHEMA:
{concept_schema}

SEMANTIC ANALYSIS REQUIREMENTS:

1. DOCUMENT SUMMARIZATION:
   Create a concise but comprehensive summary covering:
   - Main legal issues and subject matter
   - Key parties and their roles
   - Primary legal actions or proceedings
   - Important dates and deadlines
   - Core legal arguments or positions
   - Outcomes or current status

2. KEY TOPIC IDENTIFICATION:
   Identify and classify the most important legal topics:
   - Legal practice areas (constitutional, criminal, civil, etc.)
   - Procedural aspects (motions, hearings, appeals, etc.)
   - Substantive legal issues
   - Factual matters of legal significance
   - Jurisdictional and venue considerations

3. LEGAL CONCEPT EXTRACTION:
   Extract specific legal concepts using the ontology:
   - Claims and legal assertions
   - Rules and legal authorities
   - Evidence and supporting materials
   - Applications of law to facts
   - Conclusions and legal outcomes

4. CONTENT CLASSIFICATION:
   Classify the document by:
   - Document type and purpose
   - Legal domain and practice area
   - Procedural stage or context
   - Audience and intended use
   - Complexity and sophistication level

DOCUMENT TO ANALYZE:
{document_content}

ENTITY CONTEXT:
{entities_context}

ANALYSIS INSTRUCTIONS:
- Focus on legal significance and implications
- Identify technical legal terminology and concepts
- Consider procedural and substantive law aspects
- Assess document completeness and coherence
- Note any unusual or significant legal elements

Return analysis in structured JSON format:
{{
    "document_summary": "Comprehensive summary of the legal document covering main issues, parties, actions, and outcomes (max {summary_max_length} words)",
    "key_topics": [
        {{
            "topic": "Topic name",
            "category": "practice_area|procedural|substantive|factual|jurisdictional",
            "description": "Detailed description of the topic",
            "relevance": "high|medium|low",
            "legal_significance": "Explanation of why this topic is legally significant",
            "confidence": 0.9
        }}
    ],
    "legal_concepts": [
        {{
            "concept": "Legal concept name",
            "type": "claim|rule|evidence|application|conclusion|violation|sanction",
            "description": "Detailed description of the concept",
            "context": "How this concept appears in the document",
            "ontology_mapping": "Corresponding ontology entity type",
            "confidence": 0.85
        }}
    ],
    "content_classification": {{
        "document_type": "brief|motion|opinion|contract|statute|regulation|pleading|order",
        "practice_area": "constitutional|criminal|civil|administrative|corporate|etc",
        "procedural_stage": "pre_trial|trial|post_trial|appellate|etc",
        "complexity_level": "basic|intermediate|advanced|expert",
        "target_audience": "court|opposing_counsel|client|public|etc",
        "classification_confidence": 0.8
    }},
    "semantic_metadata": {{
        "total_legal_terms": 25,
        "technical_complexity": "high|medium|low",
        "argument_structure": "well_structured|partially_structured|poorly_structured",
        "citation_density": "high|medium|low",
        "factual_detail_level": "extensive|moderate|minimal",
        "writing_quality": "excellent|good|fair|poor"
    }},
    "overall_confidence": 0.83,
    "analysis_notes": "Additional observations about semantic characteristics"
}}

Ensure high-quality analysis with confidence ≥{min_confidence}. Focus on legal accuracy and practical utility."""
        
        # Performance tracking
        self.analysis_stats = {
            "total_analyses": 0,
            "avg_confidence": 0.0,
            "avg_topics_identified": 0.0,
            "avg_concepts_extracted": 0.0,
            "avg_summary_length": 0.0,
            "processing_time_avg": 0.0,
            "document_types_analyzed": {},
            "practice_areas_seen": {}
        }

    async def analyze_document_semantics(
        self,
        document_content: str,
        entities_context: List[Dict[str, Any]] = None,
        document_metadata: Dict[str, Any] = None
    ) -> SemanticAnalysisResult:
        """
        Main semantic analysis method.
        
        Args:
            document_content: Document text to analyze
            entities_context: Extracted entities for context
            document_metadata: Document metadata for additional context
            
        Returns:
            SemanticAnalysisResult with comprehensive semantic analysis
        """
        start_time = datetime.now()
        
        try:
            # Assess analysis complexity and select model
            complexity = self._assess_semantic_complexity(document_content)
            model_config = await self.model_switcher.get_optimal_model(complexity)
            
            self.logger.info(f"Starting semantic analysis with {model_config['model']} for complexity {complexity}")
            
            # Prepare analysis context
            concept_schema = self._build_concept_schema()
            entities_json = json.dumps(entities_context[:10], indent=2) if entities_context else "None available"
            
            # Build analysis prompt
            prompt = self.semantic_prompt_template.format(
                concept_schema=concept_schema,
                document_content=self._trim_content(document_content, 5000),
                entities_context=entities_json,
                summary_max_length=self.summary_max_length,
                min_confidence=self.min_confidence_threshold
            )
            
            # Perform semantic analysis
            response = await self.llm_manager.query(
                prompt=prompt,
                model=model_config['model'],
                provider=model_config['provider'],
                temperature=0.3,  # Moderate temperature for creative summarization
                max_tokens=3500
            )
            
            # Parse and validate analysis results
            analysis_data = self._parse_semantic_response(response.content)
            
            # Calculate metrics
            confidence_score = analysis_data.get('overall_confidence', 0.0)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = SemanticAnalysisResult(
                document_summary=analysis_data.get('document_summary', ''),
                key_topics=analysis_data.get('key_topics', []),
                legal_concepts=analysis_data.get('legal_concepts', []),
                content_classification=analysis_data.get('content_classification', {}),
                semantic_metadata=analysis_data.get('semantic_metadata', {}),
                confidence_score=confidence_score,
                processing_time=processing_time,
                model_used=model_config['model']
            )
            
            # Update performance statistics
            self._update_analysis_stats(result)
            
            # Log analysis summary
            self.logger.info(f"Semantic analysis completed: "
                           f"summary length: {len(result.document_summary)} chars, "
                           f"{len(result.key_topics)} topics, "
                           f"{len(result.legal_concepts)} concepts, "
                           f"confidence: {confidence_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Semantic analysis failed: {str(e)}")
            await self._record_error("semantic_analysis_failed", {"error": str(e)})
            raise

    def _build_concept_schema(self) -> str:
        """Build legal concept schema for semantic analysis guidance."""
        
        # Core semantic analysis concepts
        semantic_concepts = [
            "LEGAL_ISSUE", "RULE", "APPLICATION", "CONCLUSION", "EVIDENCE", 
            "CLAIM", "VIOLATION", "SANCTION", "PRECEDENT", "STATUTE"
        ]
        
        schema_lines = []
        for concept in semantic_concepts:
            try:
                entity_type = getattr(LegalEntityType, concept, None)
                if entity_type:
                    schema_lines.append(f"- {concept}: {entity_type.prompt_hint}")
                else:
                    # Fallback descriptions for missing ontology entries
                    fallback_descriptions = {
                        "LEGAL_ISSUE": "Legal questions or disputes requiring resolution",
                        "RULE": "Legal principles, statutes, regulations, or precedents",
                        "APPLICATION": "How legal rules apply to specific facts",
                        "CONCLUSION": "Legal outcomes, decisions, or recommendations",
                        "EVIDENCE": "Facts, documents, or testimony supporting claims",
                        "CLAIM": "Legal assertions or demands made by parties",
                        "VIOLATION": "Breach of legal duty or standard",
                        "SANCTION": "Penalty or consequence for violations",
                        "PRECEDENT": "Prior judicial decisions with binding authority",
                        "STATUTE": "Written laws enacted by legislative bodies"
                    }
                    if concept in fallback_descriptions:
                        schema_lines.append(f"- {concept}: {fallback_descriptions[concept]}")
            except AttributeError:
                continue
        
        return '\n'.join(schema_lines)

    def _parse_semantic_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM response into structured semantic analysis data."""
        
        try:
            # Handle JSON markdown blocks
            if '```json' in response_content:
                json_content = response_content.split('```json')[1].split('```')[0]
            elif '```' in response_content:
                json_content = response_content.split('```')[1].split('```')[0]
            else:
                json_content = response_content
            
            analysis_data = json.loads(json_content.strip())
            
            # Validate and normalize structure
            validated_data = {
                'document_summary': analysis_data.get('document_summary', ''),
                'key_topics': self._validate_topics(analysis_data.get('key_topics', [])),
                'legal_concepts': self._validate_legal_concepts(analysis_data.get('legal_concepts', [])),
                'content_classification': self._validate_content_classification(
                    analysis_data.get('content_classification', {})
                ),
                'semantic_metadata': self._validate_semantic_metadata(
                    analysis_data.get('semantic_metadata', {})
                ),
                'overall_confidence': float(analysis_data.get('overall_confidence', 0.0)),
                'analysis_notes': analysis_data.get('analysis_notes', '')
            }
            
            return validated_data
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.logger.warning(f"Failed to parse semantic response: {str(e)}")
            
            # Return minimal structure on parsing failure
            return {
                'document_summary': 'Semantic analysis parsing failed',
                'key_topics': [],
                'legal_concepts': [],
                'content_classification': {'document_type': 'unknown'},
                'semantic_metadata': {},
                'overall_confidence': 0.0,
                'analysis_notes': f"Response parsing error: {str(e)}"
            }

    def _validate_topics(self, topics_data: List[Any]) -> List[Dict[str, Any]]:
        """Validate and normalize topic data."""
        validated_topics = []
        
        for item in topics_data:
            if not isinstance(item, dict):
                continue
                
            topic = {
                'topic': item.get('topic', ''),
                'category': item.get('category', 'unknown'),
                'description': item.get('description', ''),
                'relevance': item.get('relevance', 'low'),
                'legal_significance': item.get('legal_significance', ''),
                'confidence': float(item.get('confidence', 0.0))
            }
            
            # Only include topics with meaningful content and confidence
            if (topic['topic'] and 
                topic['description'] and 
                topic['confidence'] >= self.min_confidence_threshold):
                validated_topics.append(topic)
        
        # Limit number of topics
        return validated_topics[:self.max_topics_per_analysis]

    def _validate_legal_concepts(self, concepts_data: List[Any]) -> List[Dict[str, Any]]:
        """Validate and normalize legal concept data."""
        validated_concepts = []
        
        for item in concepts_data:
            if not isinstance(item, dict):
                continue
                
            concept = {
                'concept': item.get('concept', ''),
                'type': item.get('type', 'unknown'),
                'description': item.get('description', ''),
                'context': item.get('context', ''),
                'ontology_mapping': item.get('ontology_mapping', ''),
                'confidence': float(item.get('confidence', 0.0))
            }
            
            # Only include concepts with meaningful content and confidence
            if (concept['concept'] and 
                concept['description'] and 
                concept['confidence'] >= self.min_confidence_threshold):
                validated_concepts.append(concept)
        
        return validated_concepts

    def _validate_content_classification(self, classification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize content classification data."""
        return {
            'document_type': classification_data.get('document_type', 'unknown'),
            'practice_area': classification_data.get('practice_area', 'unknown'),
            'procedural_stage': classification_data.get('procedural_stage', 'unknown'),
            'complexity_level': classification_data.get('complexity_level', 'unknown'),
            'target_audience': classification_data.get('target_audience', 'unknown'),
            'classification_confidence': float(classification_data.get('classification_confidence', 0.0))
        }

    def _validate_semantic_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize semantic metadata."""
        return {
            'total_legal_terms': int(metadata.get('total_legal_terms', 0)),
            'technical_complexity': metadata.get('technical_complexity', 'unknown'),
            'argument_structure': metadata.get('argument_structure', 'unknown'),
            'citation_density': metadata.get('citation_density', 'unknown'),
            'factual_detail_level': metadata.get('factual_detail_level', 'unknown'),
            'writing_quality': metadata.get('writing_quality', 'unknown')
        }

    def _assess_semantic_complexity(self, document_content: str) -> TaskComplexity:
        """Assess semantic analysis complexity for model selection."""
        
        content_length = len(document_content)
        
        # Base complexity on length
        if content_length < 2000:
            complexity = TaskComplexity.SIMPLE
        elif content_length > 7000:
            complexity = TaskComplexity.COMPLEX
        else:
            complexity = TaskComplexity.MODERATE
        
        # Look for semantic complexity indicators
        complex_indicators = [
            'whereas', 'therefore', 'notwithstanding', 'pursuant to',
            'constitutional', 'precedent', 'statutory', 'regulatory',
            'jurisdiction', 'venue', 'standing', 'liability',
            'damages', 'injunction', 'declaratory', 'mandamus'
        ]
        
        content_lower = document_content.lower()
        indicator_count = sum(1 for indicator in complex_indicators if indicator in content_lower)
        
        # Upgrade complexity based on legal complexity indicators
        if indicator_count >= 8:
            complexity = TaskComplexity.COMPLEX
        elif indicator_count >= 4 and complexity == TaskComplexity.SIMPLE:
            complexity = TaskComplexity.MODERATE
        
        return complexity

    def _trim_content(self, content: str, max_length: int) -> str:
        """Trim content to maximum length with ellipsis."""
        if len(content) <= max_length:
            return content
        return content[:max_length] + "... [TRUNCATED]"

    def _update_analysis_stats(self, result: SemanticAnalysisResult):
        """Update performance statistics."""
        self.analysis_stats["total_analyses"] += 1
        
        # Update rolling averages
        total = self.analysis_stats["total_analyses"]
        
        # Confidence
        old_conf = self.analysis_stats["avg_confidence"]
        self.analysis_stats["avg_confidence"] = (
            old_conf * (total - 1) + result.confidence_score
        ) / total
        
        # Topics identified
        topics_count = len(result.key_topics)
        old_topics = self.analysis_stats["avg_topics_identified"]
        self.analysis_stats["avg_topics_identified"] = (
            old_topics * (total - 1) + topics_count
        ) / total
        
        # Concepts extracted
        concepts_count = len(result.legal_concepts)
        old_concepts = self.analysis_stats["avg_concepts_extracted"]
        self.analysis_stats["avg_concepts_extracted"] = (
            old_concepts * (total - 1) + concepts_count
        ) / total
        
        # Summary length
        summary_length = len(result.document_summary)
        old_length = self.analysis_stats["avg_summary_length"]
        self.analysis_stats["avg_summary_length"] = (
            old_length * (total - 1) + summary_length
        ) / total
        
        # Processing time
        old_time = self.analysis_stats["processing_time_avg"]
        self.analysis_stats["processing_time_avg"] = (
            old_time * (total - 1) + result.processing_time
        ) / total
        
        # Document types and practice areas
        doc_type = result.content_classification.get('document_type', 'unknown')
        practice_area = result.content_classification.get('practice_area', 'unknown')
        
        if doc_type in self.analysis_stats["document_types_analyzed"]:
            self.analysis_stats["document_types_analyzed"][doc_type] += 1
        else:
            self.analysis_stats["document_types_analyzed"][doc_type] = 1
            
        if practice_area in self.analysis_stats["practice_areas_seen"]:
            self.analysis_stats["practice_areas_seen"][practice_area] += 1
        else:
            self.analysis_stats["practice_areas_seen"][practice_area] = 1

    async def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get current semantic analysis performance statistics."""
        return {
            **self.analysis_stats,
            "agent_status": await self.get_health_status(),
            "configuration": {
                "min_confidence_threshold": self.min_confidence_threshold,
                "max_topics_per_analysis": self.max_topics_per_analysis,
                "summary_max_length": self.summary_max_length,
                "include_legal_concepts": self.include_legal_concepts,
                "classify_content_type": self.classify_content_type
            }
        }

    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process semantic analysis task (implementation of abstract method)."""
        try:
            document_content = task_data.get('document_content', '')
            entities_context = task_data.get('entities_context', [])
            document_metadata = task_data.get('document_metadata', {})
            
            if not document_content:
                raise ValueError("No document content provided for semantic analysis")
            
            result = await self.analyze_document_semantics(
                document_content=document_content,
                entities_context=entities_context,
                document_metadata=document_metadata
            )
            
            return {
                "status": "success",
                "result": result.to_dict(),
                "metadata": {
                    "processing_time": result.processing_time,
                    "confidence_score": result.confidence_score,
                    "summary_length": len(result.document_summary),
                    "topics_identified": len(result.key_topics),
                    "concepts_extracted": len(result.legal_concepts),
                    "document_type": result.content_classification.get('document_type', 'unknown'),
                    "practice_area": result.content_classification.get('practice_area', 'unknown')
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "metadata": {"processing_time": 0.0}
            }

    def apply_feedback_adjustments(self, feedback: List[Dict[str, Any]]):
        """Apply feedback adjustments to improve semantic analysis quality."""
        adjustments = 0
        
        for fb in feedback:
            if fb.get("agent") != "semantic_analysis":
                continue
                
            feedback_type = fb.get("type")
            
            if feedback_type == "incomplete_summary":
                self.semantic_prompt_template = self.semantic_prompt_template.replace(
                    "Create a concise but comprehensive summary", 
                    "Create a *thorough and comprehensive* summary covering all important aspects"
                )
                self.summary_max_length = min(1500, self.summary_max_length + 200)
                self.logger.info("Feedback applied: Enhanced summary comprehensiveness")
                adjustments += 1
                
            elif feedback_type == "irrelevant_topics":
                self.semantic_prompt_template = self.semantic_prompt_template.replace(
                    "Identify and classify the most important legal topics",
                    "Identify and classify *only the most relevant and significant* legal topics"
                )
                self.max_topics_per_analysis = max(8, self.max_topics_per_analysis - 2)
                self.logger.info("Feedback applied: Emphasized topic relevance")
                adjustments += 1
                
            elif feedback_type == "missed_legal_concepts":
                self.include_legal_concepts = True
                self.semantic_prompt_template = self.semantic_prompt_template.replace(
                    "Extract specific legal concepts",
                    "Extract *ALL* specific legal concepts comprehensively"
                )
                self.logger.info("Feedback applied: Enhanced legal concept extraction")
                adjustments += 1
                
            elif feedback_type == "incorrect_classification":
                self.classify_content_type = True
                self.semantic_prompt_template = self.semantic_prompt_template.replace(
                    "Classify the document by:",
                    "Classify the document *carefully and precisely* by:"
                )
                self.logger.info("Feedback applied: Enhanced content classification")
                adjustments += 1
        
        if adjustments > 0:
            self.logger.info(f"Applied {adjustments} feedback adjustments to SemanticAnalysisAgent")