# legal_ai_system/agents/semantic_analysis/semantic_analysis_agent.py
"""
SemanticAnalysisAgent - Document summarization and legal topic identification.

Provides comprehensive semantic analysis of legal documents including summarization,
key legal topic identification, concept extraction, and contextual understanding
using legal ontology guidance.
"""

import asyncio
import json
# import logging # Replaced by detailed_logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict # Added field, asdict
from datetime import datetime, timezone # Added timezone

# Core imports from the new structure
from ..core.base_agent import BaseAgent
from ..utils.ontology import LegalEntityType, get_entity_types_for_prompt # Assuming LegalEntityType is an Enum
from ..core.llm_providers import LLMManager, LLMProviderError # Using LLMManager
from ..core.model_switcher import ModelSwitcher, TaskComplexity
from ..core.unified_exceptions import AgentExecutionError
from ..core.detailed_logging import LogCategory # For logger category

from ..core.agent_unified_config import create_agent_memory_mixin
from ..core.unified_memory_manager import MemoryType

# Create memory mixin for agents
MemoryMixin = create_agent_memory_mixin()


# Logger will be inherited from BaseAgent.

@dataclass
class SemanticAnalysisResult:
    """Results from semantic analysis of legal document."""
    document_summary: str = ""
    key_topics: List[Dict[str, Any]] = field(default_factory=list)
    legal_concepts: List[Dict[str, Any]] = field(default_factory=list)
    content_classification: Dict[str, Any] = field(default_factory=dict)
    semantic_metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    processing_time_sec: float = 0.0 # Renamed for clarity
    model_used: str = ""
    errors: List[str] = field(default_factory=list)
    analyzed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat()) # Added analyzed_at

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SemanticAnalysisAgent(BaseAgent):
    """
    Comprehensive semantic analysis agent for legal documents.
    """
    
    def __init__(self, service_container: Any, **config: Any): # Renamed services, added config
        super().__init__(service_container, name="SemanticAnalysisAgent", agent_type="analysis")
        
        
        # Get optimized Grok-Mini configuration for this agent
        self.llm_config = self.get_optimized_llm_config()
        self.logger.info(f"SemanticAnalysisAgentAgent configured with model: {self.llm_config.get('llm_model', 'default')}")
        self.llm_manager: Optional[LLMManager] = self.get_llm_manager()
        self.model_switcher: Optional[ModelSwitcher] = self._get_service("model_switcher") # BaseAgent helper

        if not self.llm_manager:
            self.logger.error("LLMManager service not available. SemanticAnalysisAgent may not function correctly.")
        if not self.model_switcher:
            self.logger.warning("ModelSwitcher service not available. Using default model selection logic.")

        # Configuration (can also be loaded via self.config from BaseAgent)
        self.min_confidence_threshold = float(config.get('min_confidence_threshold', 0.7))
        self.max_topics_per_analysis = int(config.get('max_topics_per_analysis', 15))
        self.summary_max_length_words = int(config.get('summary_max_length_words', 1000)) # Clarified unit
        # self.include_legal_concepts = bool(config.get('include_legal_concepts', True)) # Used in prompt
        # self.classify_content_type = bool(config.get('classify_content_type', True)) # Used in prompt
        
        # Semantic analysis prompt template (remains largely the same)
        self.semantic_prompt_template = """Perform comprehensive semantic analysis of the legal document using legal ontology concepts.
LEGAL CONCEPT SCHEMA:
{concept_schema}
SEMANTIC ANALYSIS REQUIREMENTS:
1. DOCUMENT SUMMARIZATION: Create a concise but comprehensive summary... (max {summary_max_length_words} words)
2. KEY TOPIC IDENTIFICATION: Identify and classify the most important legal topics...
3. LEGAL CONCEPT EXTRACTION: Extract specific legal concepts using the ontology...
4. CONTENT CLASSIFICATION: Classify the document by...
DOCUMENT TO ANALYZE:
{document_content}
ENTITY CONTEXT:
{entities_context}
ANALYSIS INSTRUCTIONS: ...
Return analysis in structured JSON format:
{{
    "document_summary": "...",
    "key_topics": [{{...}}],
    "legal_concepts": [{{...}}],
    "content_classification": {{...}},
    "semantic_metadata": {{...}},
    "overall_confidence": 0.83,
    "analysis_notes": "..."
}}
Ensure high-quality analysis with confidence ≥{min_confidence}. Focus on legal accuracy and practical utility."""
        
        # Performance tracking (BaseAgent has basic stats, this can be for more specific ones)
        self.analysis_stats = {
            "total_analyses": 0, "avg_confidence": 0.0, "avg_topics_identified": 0.0,
            "avg_concepts_extracted": 0.0, "avg_summary_length_chars": 0.0, # Clarified unit
            "processing_time_avg_sec": 0.0, "document_types_analyzed": {}, "practice_areas_seen": {}
        }
        self.logger.info("SemanticAnalysisAgent initialized.", parameters=self.get_config_summary_params())

    def get_config_summary_params(self) -> Dict[str, Any]:
        return {
            'min_conf': self.min_confidence_threshold, 'max_topics': self.max_topics_per_analysis,
            'summary_len': self.summary_max_length_words
        }

    async def _process_task(self, task_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing logic called by BaseAgent.process.
        task_data is expected to contain: document_content, entities_context, document_metadata.
        """
        self.logger.info("Starting semantic analysis task.", parameters={'doc_id': metadata.get("document_id", "unknown")})
        start_time_obj = datetime.now(timezone.utc)
        
        document_content = task_data.get('document_content', '')
        entities_context = task_data.get('entities_context', [])
        # document_metadata_param = task_data.get('document_metadata', {}) # Renamed to avoid conflict

        if not document_content:
            self.logger.error("No document content provided for semantic analysis.")
            return SemanticAnalysisResult(errors=["No document content provided."]).to_dict()
        if not self.llm_manager:
            self.logger.error("LLMManager not available, cannot perform semantic analysis.")
            return SemanticAnalysisResult(errors=["LLMManager not available."]).to_dict()

        try:
            complexity = self._assess_semantic_complexity(document_content)
            # Use ModelSwitcher if available, otherwise use primary_config model
            model_to_use = self.llm_manager.primary_config.model
            provider_to_use = self.llm_manager.primary_config.provider.value
            
            if self.model_switcher:
                # model_config_dict = await self.model_switcher.get_optimal_model(complexity) # Assuming this method exists
                # For now, let's assume ModelSwitcher suggests a model name, and we use primary provider
                suggested_model_name = self.model_switcher.suggest_model_for_task("semantic_analysis", complexity)
                if suggested_model_name: model_to_use = suggested_model_name
                # Provider could also be part of model_config_dict if ModelSwitcher handles multi-provider suggestions
            
            self.logger.info(f"Semantic analysis with model.", parameters={'model': model_to_use, 'provider': provider_to_use, 'complexity': complexity.value})
            
            concept_schema = self._build_concept_schema()
            entities_json = json.dumps(entities_context[:10], indent=2) if entities_context else "None available"
            
            prompt = self.semantic_prompt_template.format(
                concept_schema=concept_schema,
                document_content=self._trim_content(document_content, 5000), # Max length for prompt
                entities_context=entities_json,
                summary_max_length_words=self.summary_max_length_words,
                min_confidence=self.min_confidence_threshold
            )
            
            llm_response_obj = await self.llm_manager.complete( # Call LLMManager
                prompt=prompt,
                model=model_to_use, # Pass model to use
                provider=LLMProviderEnum(provider_to_use), # Pass provider
                temperature=0.3, max_tokens=3500 # Example params
            )
            
            analysis_data = self._parse_semantic_response(llm_response_obj.content)
            
            confidence_score = analysis_data.get('overall_confidence', 0.0)
            processing_time_sec = (datetime.now(timezone.utc) - start_time_obj).total_seconds()
            
            result = SemanticAnalysisResult(
                document_summary=analysis_data.get('document_summary', ''),
                key_topics=analysis_data.get('key_topics', []),
                legal_concepts=analysis_data.get('legal_concepts', []),
                content_classification=analysis_data.get('content_classification', {}),
                semantic_metadata=analysis_data.get('semantic_metadata', {}),
                confidence_score=confidence_score,
                processing_time_sec=processing_time_sec,
                model_used=model_to_use # Or llm_response_obj.model_name
            )
            
            self._update_internal_analysis_stats(result) # Renamed
            self.logger.info("Semantic analysis task completed.", 
                            parameters={'doc_id': metadata.get("document_id", "unknown"), 'confidence': confidence_score, 'topics': len(result.key_topics)})
            return result.to_dict()
            
        except LLMProviderError as e:
            self.logger.error("LLMProviderError during semantic analysis.", exception=e)
            return SemanticAnalysisResult(errors=[f"LLM Error: {str(e)}"], model_used=model_to_use).to_dict()
        except Exception as e:
            self.logger.error("Unexpected error during semantic analysis.", exception=e)
            # Consider raising AgentExecutionError(str(e), cause=e) if BaseAgent handles it
            return SemanticAnalysisResult(errors=[f"Unexpected error: {str(e)}"], model_used=model_to_use).to_dict()

    def _build_concept_schema(self) -> str:
        """Build legal concept schema for semantic analysis guidance."""
        # ... (logic remains similar, ensure LegalEntityType enum is accessible)
        semantic_concepts = [
            "LEGAL_ISSUE", "RULE", "APPLICATION", "CONCLUSION", "EVIDENCE", 
            "CLAIM", "VIOLATION", "SANCTION", "PRECEDENT", "STATUTE"
        ]
        schema_lines = []
        if LegalEntityType: # Check if ontology was loaded
            for concept_name_str in semantic_concepts:
                try:
                    entity_type_enum_member = getattr(LegalEntityType, concept_name_str, None)
                    if entity_type_enum_member:
                        # Access .value (EntityMeta) then .prompt_hint
                        schema_lines.append(f"- {concept_name_str}: {entity_type_enum_member.value.prompt_hint}")
                    # else: fallback descriptions if needed, as in original
                except AttributeError: # If LegalEntityType doesn't have the concept or .value.prompt_hint structure
                    self.logger.warning(f"Attribute error building concept schema for {concept_name_str}.")
                    continue # Or add a default hint
        else:
            self.logger.warning("LegalEntityType ontology not available for building concept schema.")
            # Provide very basic fallback hints if ontology is missing
            schema_lines = [f"- {concept}: A type of legal concept." for concept in semantic_concepts]

        return '\n'.join(schema_lines)

    def _parse_semantic_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM response into structured semantic analysis data."""
        # ... (logic remains similar, ensure robust JSON parsing)
        try:
            json_content = response_content # Assume LLM returns clean JSON
            # More robust: try to extract JSON from markdown blocks if present
            if '```json' in response_content:
                json_content = response_content.split('```json')[1].split('```')
            elif '```' in response_content and response_content.strip().startswith('```') and response_content.strip().endswith('```'):
                 json_content = response_content.strip()[3:-3] # Remove triple backticks
            
            parsed_data = json.loads(json_content.strip())
            
            # Basic validation and normalization
            validated_data = {
                'document_summary': parsed_data.get('document_summary', ''),
                'key_topics': self._validate_list_of_dicts(parsed_data.get('key_topics', [])),
                'legal_concepts': self._validate_list_of_dicts(parsed_data.get('legal_concepts', [])),
                'content_classification': parsed_data.get('content_classification', {}),
                'semantic_metadata': parsed_data.get('semantic_metadata', {}),
                'overall_confidence': float(parsed_data.get('overall_confidence', 0.0)),
                'analysis_notes': parsed_data.get('analysis_notes', '')
            }
            return validated_data
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.logger.warning(f"Failed to parse LLM semantic response. Content: {response_content[:200]}...", exception=e)
            return { # Return default structure on error
                'document_summary': 'Error: Could not parse LLM analysis.', 'key_topics': [], 'legal_concepts': [],
                'content_classification': {}, 'semantic_metadata': {}, 'overall_confidence': 0.0,
                'analysis_notes': f"Response parsing error: {str(e)}"
            }

    def _validate_list_of_dicts(self, data_list: Any) -> List[Dict[str, Any]]:
        """Helper to ensure a list contains dictionaries."""
        if not isinstance(data_list, list): return []
        return [item for item in data_list if isinstance(item, dict)]

    # _validate_topics, _validate_legal_concepts, etc. from original can be used here
    # for more detailed validation of the parsed structure.

    def _assess_semantic_complexity(self, document_content: str) -> TaskComplexity:
        """Assess semantic analysis complexity for model selection."""
        # ... (logic remains similar)
        content_length = len(document_content)
        if content_length < 2000: complexity = TaskComplexity.SIMPLE
        elif content_length > 7000: complexity = TaskComplexity.COMPLEX
        else: complexity = TaskComplexity.MODERATE
        
        # Further refinement based on keyword density could be added
        return complexity

    def _trim_content(self, content: str, max_length_chars: int) -> str: # Clarified unit
        """Trim content to maximum length with ellipsis."""
        if len(content) <= max_length_chars: return content
        return content[:max_length_chars - len("... [TRUNCATED]")] + "... [TRUNCATED]"

    def _update_internal_analysis_stats(self, result: SemanticAnalysisResult): # Renamed
        """Update internal performance statistics for this agent."""
        # ... (logic largely remains, ensure keys match SemanticAnalysisResult and self.analysis_stats)
        self.analysis_stats["total_analyses"] += 1
        total = self.analysis_stats["total_analyses"]
        
        # Confidence
        self.analysis_stats["avg_confidence"] = (self.analysis_stats["avg_confidence"] * (total - 1) + result.confidence_score) / total
        # Topics identified
        self.analysis_stats["avg_topics_identified"] = (self.analysis_stats["avg_topics_identified"] * (total - 1) + len(result.key_topics)) / total
        # Concepts extracted
        self.analysis_stats["avg_concepts_extracted"] = (self.analysis_stats["avg_concepts_extracted"] * (total - 1) + len(result.legal_concepts)) / total
        # Summary length
        self.analysis_stats["avg_summary_length_chars"] = (self.analysis_stats["avg_summary_length_chars"] * (total - 1) + len(result.document_summary)) / total
        # Processing time
        self.analysis_stats["processing_time_avg_sec"] = (self.analysis_stats["processing_time_avg_sec"] * (total - 1) + result.processing_time_sec) / total
        
        doc_type = result.content_classification.get('document_type', 'unknown')
        self.analysis_stats["document_types_analyzed"][doc_type] = self.analysis_stats["document_types_analyzed"].get(doc_type, 0) + 1
        # ... similar for practice_areas_seen

    async def get_analysis_statistics(self) -> Dict[str, Any]: # This is a public method for stats
        """Get current semantic analysis performance statistics."""
        # ... (logic remains similar)
        health = await self.health_check() # Get base health
        return {
            **self.analysis_stats,
            "agent_health_status": health, # Renamed from agent_status
            "current_config": self.get_config_summary_params() # Renamed from configuration
        }
    
    # apply_feedback_adjustments from original can be kept if feedback mechanism is implemented.
    # For now, it modifies self.semantic_prompt_template and config values.