# legal_ai_system/agents/legal_analysis/legal_analysis_agent.py
"""
LegalAnalysisAgent - IRAC analysis with ontology alignment and contradiction detection.

Performs comprehensive legal analysis using the IRAC framework (Issue, Rule, Application, Conclusion)
enhanced with contradiction detection, causal chain analysis, and legal reasoning validation.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

# Core imports from the new structure
from ..core.base_agent import BaseAgent
from ..utils.ontology import LegalEntityType  # Assuming LegalEntityType is an Enum
from ..core.llm_providers import LLMManager, LLMProviderError, LLMProviderEnum
from ..core.model_switcher import ModelSwitcher, TaskComplexity
from ..core.agent_unified_config import create_agent_memory_mixin

# Create memory mixin for agents
MemoryMixin = create_agent_memory_mixin()

# Logger will be inherited from BaseAgent.

@dataclass
class LegalAnalysisResult:
    """Results from comprehensive legal analysis."""
    irac_summary: Dict[str, Any] = field(default_factory=lambda: {"issues": [], "rules": [], "application": "", "conclusion": ""}) # Ensure keys
    contradictions_found: List[Dict[str, Any]] = field(default_factory=list) # Renamed
    causal_chains_identified: List[Dict[str, Any]] = field(default_factory=list) # Renamed
    extracted_legal_concepts: List[Dict[str, Any]] = field(default_factory=list) # Renamed
    confidence_score: float = 0.0
    processing_time_sec: float = 0.0 # Renamed
    model_used: str = ""
    analysis_depth_level: str = TaskComplexity.MODERATE.value # Renamed, use Enum value
    errors: List[str] = field(default_factory=list)
    analyzed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat()) # Added

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LegalAnalysisAgent(BaseAgent, MemoryMixin):
    """
    Advanced legal analysis agent using IRAC framework with ontology alignment.
    Enhanced with Grok-Mini LLM integration and shared memory capabilities.
    """
    
    def __init__(self, service_container: Any, **config: Any): # Renamed services, added config
        super().__init__(service_container, name="LegalAnalysisAgent", agent_type="advanced_analysis")
        
        self.llm_manager: Optional[LLMManager] = self.get_llm_manager()
        self.model_switcher: Optional[ModelSwitcher] = self._get_service("model_switcher")

        if not self.llm_manager:
            self.logger.error("LLMManager service not available. LegalAnalysisAgent may not function fully.")
        
        # Get optimized Grok-Mini configuration for this agent
        self.llm_config = self.get_optimized_llm_config()
        self.logger.info(f"LegalAnalysisAgent configured with model: {self.llm_config.get('llm_model', 'default')}")

        # Analysis configuration
        self.min_confidence_threshold = float(config.get('min_confidence_threshold', 0.7))
        # self.max_analysis_length_chars = int(config.get('max_analysis_length_chars', 8000)) # Renamed, for input trimming
        # self.include_precedent_analysis = bool(config.get('include_precedent_analysis', True)) # Used in prompt logic
        # self.detect_contradictions = bool(config.get('detect_contradictions', True)) # Used in prompt logic
        # self.analyze_causal_chains = bool(config.get('analyze_causal_chains', True)) # Used in prompt logic
        
        # Enhanced IRAC analysis prompt template (remains largely the same)
        self.analysis_prompt_template = """Perform comprehensive legal analysis using the IRAC framework and ontology-aligned concepts.
LEGAL ONTOLOGY CONCEPTS:
{ontology_hints}
ANALYSIS FRAMEWORK: ...
ENHANCED ANALYSIS REQUIREMENTS: ...
DOCUMENT CONTENT:
{document_content}
SEMANTIC CONTEXT:
{semantic_context}
STRUCTURAL ANALYSIS:
{structural_context}
EXTRACTED ENTITIES:
{entities_context}
Return analysis in structured JSON format:
{{
    "irac_summary": {{ "issues": [], "rules": [], "application": "", "conclusion": "" }},
    "contradictions": [{{...}}],
    "causal_chains": [{{...}}],
    "legal_concepts": [{{...}}],
    "overall_confidence": 0.85,
    "analysis_notes": "..."
}}
Ensure thorough analysis with high confidence scores (≥{min_confidence}). Focus on legal accuracy and practical implications."""
        
        # Performance tracking
        self.analysis_stats = { # Agent-specific stats
            "total_analyses_run": 0, "avg_confidence": 0.0, "avg_issues_identified": 0.0, # Renamed
            "avg_contradictions_found": 0.0, "avg_causal_chains_found": 0.0, "processing_time_avg_sec": 0.0
        }
        self.logger.info("LegalAnalysisAgent initialized.", parameters=self.get_config_summary_params())

    def get_config_summary_params(self) -> Dict[str,Any]:
        return {
            'min_conf': self.min_confidence_threshold,
            # 'max_len_chars': self.max_analysis_length_chars,
        }

    async def _process_task(self, task_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing logic.
        task_data: document_content, semantic_context, structural_context, entities_context, document_metadata
        """
        self.logger.info("Starting legal analysis task.", parameters={'doc_id': metadata.get("document_id", "unknown")})
        start_time_obj = datetime.now(timezone.utc)

        document_content = task_data.get('document_content', '')
        semantic_context_str = task_data.get('semantic_context', '') # Renamed
        structural_context_str = task_data.get('structural_context', '') # Renamed
        entities_context_list = task_data.get('entities_context', []) # Renamed
        # document_metadata_param = task_data.get('document_metadata', {})

        if not document_content:
            self.logger.error("No document content provided for legal analysis.")
            return LegalAnalysisResult(errors=["No document content provided."]).to_dict()
        if not self.llm_manager:
            self.logger.error("LLMManager not available, cannot perform legal analysis.")
            return LegalAnalysisResult(errors=["LLMManager not available."]).to_dict()

        model_to_use = self.llm_manager.primary_config.model # Default
        try:
            complexity = self._assess_analysis_complexity(document_content, semantic_context_str, structural_context_str)
            provider_to_use = self.llm_manager.primary_config.provider.value # Default
            if self.model_switcher:
                suggested_model_name = self.model_switcher.suggest_model_for_task("legal_analysis", complexity)
                if suggested_model_name: model_to_use = suggested_model_name
            
            self.logger.info(f"Legal analysis with model.", parameters={'model': model_to_use, 'provider': provider_to_use, 'complexity': complexity.value})

            ontology_hints = self._build_ontology_hints()
            entities_json = json.dumps(entities_context_list[:10], indent=2, default=str) if entities_context_list else "None available"
            
            prompt = self.analysis_prompt_template.format(
                ontology_hints=ontology_hints,
                document_content=self._trim_content(document_content, self.config.get('max_analysis_length_chars', 8000) * 0.5), # Trim main content more
                semantic_context=self._trim_content(semantic_context_str, 1500),
                structural_context=self._trim_content(structural_context_str, 1500),
                entities_context=entities_json,
                min_confidence=self.min_confidence_threshold
            )
            
            llm_response_obj = await self.llm_manager.complete(
                prompt=prompt, model=model_to_use, provider=LLMProviderEnum(provider_to_use),
                temperature=0.2, max_tokens=4000 # Max tokens might need adjustment based on expected output size
            )
            
            analysis_data = self._parse_analysis_response(llm_response_obj.content)
            
            confidence_score_val = analysis_data.get('overall_confidence', 0.0) # Renamed
            processing_time_val = (datetime.now(timezone.utc) - start_time_obj).total_seconds() # Renamed
            
            result = LegalAnalysisResult(
                irac_summary=analysis_data.get('irac_summary', {}),
                contradictions_found=analysis_data.get('contradictions', []),
                causal_chains_identified=analysis_data.get('causal_chains', []),
                extracted_legal_concepts=analysis_data.get('legal_concepts', []),
                confidence_score=confidence_score_val,
                processing_time_sec=processing_time_val,
                model_used=model_to_use, # Or llm_response_obj.model_name
                analysis_depth_level=complexity.value
            )
            
            self._update_internal_analysis_stats(result) # Renamed
            self.logger.info("Legal analysis task completed.", 
                            parameters={'doc_id': metadata.get("document_id", "unknown"), 'confidence': confidence_score_val,
                                        'issues': len(result.irac_summary.get('issues',[]))})
            return result.to_dict()

        except LLMProviderError as e:
            self.logger.error("LLMProviderError during legal analysis.", exception=e)
            return LegalAnalysisResult(errors=[f"LLM Error: {str(e)}"], model_used=model_to_use).to_dict()
        except Exception as e:
            self.logger.error("Unexpected error during legal analysis.", exception=e)
            return LegalAnalysisResult(errors=[f"Unexpected error: {str(e)}"], model_used=model_to_use).to_dict()

    def _build_ontology_hints(self) -> str:
        """Build ontology hints for legal analysis guidance."""
        # ... (logic remains similar, ensure LegalEntityType is accessible)
        analysis_concepts_str_list = [
            "CLAIM", "RULE", "APPLICATION", "CONCLUSION", "EVIDENCE", 
            "VIOLATION", "MISCONDUCT_INCIDENT", "SANCTION", "LEGAL_ISSUE"
        ]
        hints_list: List[str] = [] # Renamed
        if LegalEntityType:
            for concept_name in analysis_concepts_str_list:
                try:
                    entity_type_enum = getattr(LegalEntityType, concept_name, None)
                    hint_text = entity_type_enum.value.prompt_hint if entity_type_enum else f"Details about {concept_name.lower().replace('_', ' ')}."
                    hints_list.append(f"- {concept_name}: {hint_text}")
                except AttributeError:
                    hints_list.append(f"- {concept_name}: General legal concept.")
        else:
            self.logger.warning("LegalEntityType ontology not available for building analysis hints.")
            hints_list = [f"- {concept}: A type of legal concept or finding." for concept in analysis_concepts_str_list]
        return '\n'.join(hints_list)

    def _parse_analysis_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM response into structured analysis data."""
        # ... (logic remains similar, ensure robust JSON parsing)
        try:
            json_content = response_content
            if '```json' in response_content:
                json_content = response_content.split('```json')[1].split('```')[0]
            elif '```' in response_content and response_content.strip().startswith('```') and response_content.strip().endswith('```'):
                json_content = response_content.strip()[3:-3]
            
            parsed_data = json.loads(json_content.strip())
            
            # Basic validation and normalization
            validated_data = {
                'irac_summary': parsed_data.get('irac_summary', {}),
                'contradictions': self._validate_list_of_dicts(parsed_data.get('contradictions', [])),
                'causal_chains': self._validate_list_of_dicts(parsed_data.get('causal_chains', [])),
                'legal_concepts': self._validate_list_of_dicts(parsed_data.get('legal_concepts', [])),
                'overall_confidence': float(parsed_data.get('overall_confidence', 0.0)),
                'analysis_notes': parsed_data.get('analysis_notes', '')
            }
            # Ensure IRAC summary has default components
            irac = validated_data['irac_summary']
            if not isinstance(irac, dict): irac = {}; validated_data['irac_summary'] = irac
            for comp in ['issues', 'rules', 'application', 'conclusion']:
                if comp not in irac: irac[comp] = [] if comp in ['issues', 'rules'] else ""
            
            return validated_data
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.logger.warning(f"Failed to parse LLM analysis response. Content: {response_content[:200]}...", exception=e)
            return { # Return default structure
                'irac_summary': {'issues': [], 'rules': [], 'application': "", 'conclusion': ""},
                'contradictions': [], 'causal_chains': [], 'legal_concepts': [], 'overall_confidence': 0.0,
                'analysis_notes': f"Response parsing error: {str(e)}"
            }

    def _validate_list_of_dicts(self, data_list: Any) -> List[Dict[str, Any]]:
        """Helper to ensure a list contains dictionaries."""
        if not isinstance(data_list, list): return []
        return [item for item in data_list if isinstance(item, dict)]
        
    # _validate_contradictions, _validate_causal_chains, etc. can be added for more detail.

    def _assess_analysis_complexity(
        self, document_content: str, semantic_context_str: str, structural_context_str: str
    ) -> TaskComplexity:
        """Assess legal analysis complexity for model selection."""
        # ... (logic remains similar)
        total_len = len(document_content) + len(semantic_context_str) + len(structural_context_str)
        if total_len < 2000: complexity = TaskComplexity.SIMPLE
        elif total_len > 8000: complexity = TaskComplexity.COMPLEX
        else: complexity = TaskComplexity.MODERATE
        return complexity

    def _trim_content(self, content: str, max_length_chars: int) -> str: # Clarified unit
        """Trim content to maximum length with ellipsis."""
        if len(content) <= max_length_chars: return content
        return content[:max_length_chars - len("... [TRUNCATED]")] + "... [TRUNCATED]"

    def _update_internal_analysis_stats(self, result: LegalAnalysisResult): # Renamed
        """Update internal performance statistics for this agent."""
        # ... (logic largely remains)
        self.analysis_stats["total_analyses_run"] += 1
        total = self.analysis_stats["total_analyses_run"]
        self.analysis_stats["avg_confidence"] = (self.analysis_stats["avg_confidence"] * (total-1) + result.confidence_score) / total if total > 0 else result.confidence_score
        self.analysis_stats["avg_issues_identified"] = (self.analysis_stats["avg_issues_identified"] * (total-1) + len(result.irac_summary.get('issues',[]))) / total if total > 0 else len(result.irac_summary.get('issues',[]))
        self.analysis_stats["avg_contradictions_found"] = (self.analysis_stats["avg_contradictions_found"] * (total-1) + len(result.contradictions_found)) / total if total > 0 else len(result.contradictions_found)
        self.analysis_stats["avg_causal_chains_found"] = (self.analysis_stats["avg_causal_chains_found"] * (total-1) + len(result.causal_chains_identified)) / total if total > 0 else len(result.causal_chains_identified)
        self.analysis_stats["processing_time_avg_sec"] = (self.analysis_stats["processing_time_avg_sec"] * (total-1) + result.processing_time_sec) / total if total > 0 else result.processing_time_sec

    async def get_analysis_statistics(self) -> Dict[str, Any]: # Public method
        """Get current analysis performance statistics."""
        health = await self.health_check()
        return {
            **self.analysis_stats,
            "agent_health_status": health,
            "current_config": self.get_config_summary_params()
        }