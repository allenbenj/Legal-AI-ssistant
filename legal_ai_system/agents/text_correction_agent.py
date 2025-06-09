# legal_ai_system/agents/text_correction/text_correction_agent.py
"""
TextCorrectionAgent - Legal document formatting and grammar correction.

Provides comprehensive text correction services for legal documents including
grammar correction, tone adjustment, role-based formatting, and legal writing
style enhancement with context-aware improvements.
"""

import asyncio
import json
# import logging # Replaced by detailed_logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict # Added field, asdict
from datetime import datetime, timezone # Added timezone

# Core imports from the new structure
from ...core.base_agent import BaseAgent
from ...utils.ontology import LegalEntityType # Assuming LegalEntityType is an Enum for role schema hints
from ...core.llm_providers import LLMManager, LLMProviderError # Using LLMManager
from ...core.model_switcher import ModelSwitcher, TaskComplexity
from ...core.unified_exceptions import AgentExecutionError
from ...core.detailed_logging import LogCategory # For logger category

from ...config.agent_unified_config import create_agent_memory_mixin
from ...memory.unified_memory_manager import MemoryType

# Create memory mixin for agents
MemoryMixin = create_agent_memory_mixin()


# Logger will be inherited from BaseAgent.

@dataclass
class TextCorrectionResult:
    """Results from text correction analysis."""
    corrected_text: str = ""
    corrections_made: List[Dict[str, Any]] = field(default_factory=list)
    formatting_improvements: List[Dict[str, Any]] = field(default_factory=list)
    style_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    processing_time_sec: float = 0.0 # Renamed
    model_used: str = ""
    errors: List[str] = field(default_factory=list)
    corrected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat()) # Added

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TextCorrectionAgent(BaseAgent):
    """
    Advanced text correction agent for legal documents.
    """
    
    def __init__(self, service_container: Any, **config: Any): # Renamed services, added config
        super().__init__(service_container, name="TextCorrectionAgent", agent_type="correction")
        
        
        # Get optimized Grok-Mini configuration for this agent
        self.llm_config = self.get_optimized_llm_config()
        self.logger.info(f"TextCorrectionAgentAgent configured with model: {self.llm_config.get('llm_model', 'default')}")
        self.llm_manager: Optional[LLMManager] = self.get_llm_manager()
        self.model_switcher: Optional[ModelSwitcher] = self._get_service("model_switcher")

        if not self.llm_manager:
            self.logger.error("LLMManager service not available. TextCorrectionAgent may not function fully.")

        # Configuration
        self.min_confidence_threshold = float(config.get('min_confidence_threshold', 0.8))
        # self.preserve_legal_terminology = bool(config.get('preserve_legal_terminology', True)) # Used in prompt
        # self.enhance_formality = bool(config.get('enhance_formality', True)) # Used in prompt
        # self.standardize_citations = bool(config.get('standardize_citations', True)) # Used in prompt
        # self.role_based_formatting = bool(config.get('role_based_formatting', True)) # Used in prompt
        
        # Text correction prompt template (remains largely the same)
        self.correction_prompt_template = """Correct and enhance the legal text for grammar, tone, formatting, and professional legal writing standards.
LEGAL ROLE SCHEMA:
{role_schema}
CORRECTION REQUIREMENTS: ...
ORIGINAL TEXT:
{raw_text}
KNOWN ENTITIES AND CONTEXT:
{entities_context}
DOCUMENT TYPE AND CONTEXT:
{document_context}
CORRECTION INSTRUCTIONS: ...
Return corrections in structured JSON format:
{{
    "corrected_text": "...",
    "corrections_made": [{{...}}],
    "formatting_improvements": [{{...}}],
    "style_adjustments": [{{...}}],
    "quality_metrics": {{...}},
    "overall_confidence": 0.88,
    "correction_notes": "..."
}}
Ensure high-quality corrections with confidence ≥{min_confidence}. Focus on preserving legal accuracy while enhancing presentation."""
        
        # Performance tracking
        self.correction_stats = { # Agent-specific stats
            "total_corrections_run": 0, "total_errors_fixed_reported": 0, "avg_confidence": 0.0,
            "avg_improvements_reported": 0.0, "processing_time_avg_sec": 0.0,
            "correction_types_summary": defaultdict(int) # Using defaultdict
        }
        self.logger.info("TextCorrectionAgent initialized.", parameters=self.get_config_summary_params())

    def get_config_summary_params(self) -> Dict[str, Any]:
        return {
            'min_conf': self.min_confidence_threshold,
            # Add other key config params here
        }

    async def _process_task(self, task_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing logic.
        task_data: raw_text, entities_context, document_context
        """
        self.logger.info("Starting text correction task.", parameters={'doc_id': metadata.get("document_id", "unknown")})
        start_time_obj = datetime.now(timezone.utc)

        raw_text = task_data.get('raw_text', '')
        entities_context_list = task_data.get('entities_context', []) # Renamed
        document_context_dict = task_data.get('document_context', {}) # Renamed

        if not raw_text:
            self.logger.error("No raw text provided for correction.")
            return TextCorrectionResult(errors=["No raw text provided."]).to_dict()
        if not self.llm_manager:
            self.logger.error("LLMManager not available, cannot perform text correction.")
            return TextCorrectionResult(errors=["LLMManager not available."]).to_dict()

        try:
            complexity = self._assess_correction_complexity(raw_text)
            model_to_use = self.llm_manager.primary_config.model
            provider_to_use = self.llm_manager.primary_config.provider.value
            if self.model_switcher:
                suggested_model_name = self.model_switcher.suggest_model_for_task("text_correction", complexity)
                if suggested_model_name: model_to_use = suggested_model_name
            
            self.logger.info(f"Text correction with model.", parameters={'model': model_to_use, 'provider': provider_to_use, 'complexity': complexity.value})

            role_schema = self._build_role_schema()
            entities_json = json.dumps(entities_context_list[:8], indent=2) if entities_context_list else "None available"
            doc_context_json = json.dumps(document_context_dict, indent=2) if document_context_dict else "General legal document"
            
            prompt = self.correction_prompt_template.format(
                role_schema=role_schema,
                raw_text=self._trim_content(raw_text, 4000),
                entities_context=entities_json,
                document_context=doc_context_json,
                min_confidence=self.min_confidence_threshold
            )
            
            llm_response_obj = await self.llm_manager.complete(
                prompt=prompt, model=model_to_use, provider=LLMProviderEnum(provider_to_use),
                temperature=0.2, max_tokens=4000 # Max tokens might need to be > input for corrections
            )
            
            correction_data = self._parse_correction_response(llm_response_obj.content, raw_text)
            
            confidence_score = correction_data.get('overall_confidence', 0.0)
            processing_time_sec = (datetime.now(timezone.utc) - start_time_obj).total_seconds()
            
            result = TextCorrectionResult(
                corrected_text=correction_data.get('corrected_text', raw_text),
                corrections_made=correction_data.get('corrections_made', []),
                formatting_improvements=correction_data.get('formatting_improvements', []),
                style_adjustments=correction_data.get('style_adjustments', []),
                quality_metrics=correction_data.get('quality_metrics', {}),
                confidence_score=confidence_score,
                processing_time_sec=processing_time_sec,
                model_used=model_to_use # Or llm_response_obj.model_name
            )
            
            self._update_internal_correction_stats(result) # Renamed
            self.logger.info("Text correction task completed.", 
                            parameters={'doc_id': metadata.get("document_id", "unknown"), 'confidence': confidence_score,
                                        'num_corrections': len(result.corrections_made)})
            return result.to_dict()

        except LLMProviderError as e:
            self.logger.error("LLMProviderError during text correction.", exception=e)
            # Consider if _record_error is part of BaseAgent or a local helper
            # await self._record_error("text_correction_failed_llm", {"error": str(e)}) 
            return TextCorrectionResult(errors=[f"LLM Error: {str(e)}"], model_used=model_to_use, corrected_text=raw_text).to_dict()
        except Exception as e:
            self.logger.error("Unexpected error during text correction.", exception=e)
            # await self._record_error("text_correction_failed_unexpected", {"error": str(e)})
            return TextCorrectionResult(errors=[f"Unexpected error: {str(e)}"], model_used=model_to_use, corrected_text=raw_text).to_dict()

    def _build_role_schema(self) -> str:
        """Build legal role schema for correction guidance."""
        # ... (logic remains similar, ensure LegalEntityType is accessible)
        legal_roles_str_list = ["JUDGE", "ATTORNEY", "DEFENDANT", "WITNESS", "PROSECUTOR", "DEFENSECOUNSEL"]
        role_descriptions_map = { # Fallback descriptions
            "JUDGE": "Formal judicial language with authoritative tone.",
            "ATTORNEY": "Professional legal advocacy with persuasive language.",
            # ... add others
        }
        schema_lines = []
        if LegalEntityType:
            for role_name_str in legal_roles_str_list:
                try:
                    entity_type_enum_member = getattr(LegalEntityType, role_name_str, None)
                    hint = entity_type_enum_member.value.prompt_hint if entity_type_enum_member else role_descriptions_map.get(role_name_str, f"Professional {role_name_str.lower()} language")
                    schema_lines.append(f"- {role_name_str}: {hint}")
                except AttributeError:
                     schema_lines.append(f"- {role_name_str}: {role_descriptions_map.get(role_name_str, 'Standard professional language.')}")
        else:
            self.logger.warning("LegalEntityType ontology not available for building role schema.")
            schema_lines = [f"- {role}: {desc}" for role, desc in role_descriptions_map.items()]
        return '\n'.join(schema_lines)

    def _parse_correction_response(self, response_content: str, original_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured correction data."""
        # ... (logic remains similar, ensure robust JSON parsing)
        try:
            json_content = response_content
            if '```json' in response_content:
                json_content = response_content.split('```json')[1].split('```')
            elif '```' in response_content and response_content.strip().startswith('```') and response_content.strip().endswith('```'):
                 json_content = response_content.strip()[3:-3]
            
            parsed_data = json.loads(json_content.strip())
            
            # Basic validation and normalization
            validated_data = {
                'corrected_text': parsed_data.get('corrected_text', original_text), # Fallback to original
                'corrections_made': self._validate_list_of_dicts(parsed_data.get('corrections_made', [])),
                'formatting_improvements': self._validate_list_of_dicts(parsed_data.get('formatting_improvements', [])),
                'style_adjustments': self._validate_list_of_dicts(parsed_data.get('style_adjustments', [])),
                'quality_metrics': parsed_data.get('quality_metrics', {}), # Add validation if needed
                'overall_confidence': float(parsed_data.get('overall_confidence', 0.0)),
                'correction_notes': parsed_data.get('correction_notes', '')
            }
            return validated_data
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.logger.warning(f"Failed to parse LLM correction response. Content: {response_content[:200]}...", exception=e)
            return { # Return default structure on error
                'corrected_text': original_text, 'corrections_made': [], 'formatting_improvements': [], 
                'style_adjustments': [], 'quality_metrics': {}, 'overall_confidence': 0.0,
                'correction_notes': f"Response parsing error: {str(e)}"
            }
            
    def _validate_list_of_dicts(self, data_list: Any) -> List[Dict[str, Any]]:
        """Helper to ensure a list contains dictionaries."""
        if not isinstance(data_list, list): return []
        return [item for item in data_list if isinstance(item, dict)]

    # _validate_corrections, _validate_formatting_improvements, etc. can be added for more detail.

    def _assess_correction_complexity(self, text: str) -> TaskComplexity:
        """Assess text correction complexity for model selection."""
        # ... (logic remains similar)
        text_length = len(text)
        if text_length < 1000: complexity = TaskComplexity.SIMPLE
        elif text_length > 4000: complexity = TaskComplexity.COMPLEX
        else: complexity = TaskComplexity.MODERATE
        return complexity

    def _trim_content(self, content: str, max_length_chars: int) -> str: # Clarified unit
        """Trim content to maximum length with ellipsis."""
        if len(content) <= max_length_chars: return content
        return content[:max_length_chars - len("... [TRUNCATED]")] + "... [TRUNCATED]"

    def _update_internal_correction_stats(self, result: TextCorrectionResult): # Renamed
        """Update internal performance statistics for this agent."""
        # ... (logic largely remains)
        self.correction_stats["total_corrections_run"] += 1
        total_runs = self.correction_stats["total_corrections_run"]
        
        num_improvements = len(result.corrections_made) + len(result.formatting_improvements) + len(result.style_adjustments)
        self.correction_stats["avg_improvements_reported"] = (self.correction_stats["avg_improvements_reported"] * (total_runs-1) + num_improvements) / total_runs if total_runs > 0 else num_improvements
        self.correction_stats["avg_confidence"] = (self.correction_stats["avg_confidence"] * (total_runs-1) + result.confidence_score) / total_runs if total_runs > 0 else result.confidence_score
        self.correction_stats["processing_time_avg_sec"] = (self.correction_stats["processing_time_avg_sec"] * (total_runs-1) + result.processing_time_sec) / total_runs if total_runs > 0 else result.processing_time_sec

        for corr in result.corrections_made: self.correction_stats["correction_types_summary"][corr.get('type', 'unknown')] +=1
        # Similar for formatting_improvements and style_adjustments if they have 'type'

    async def get_correction_statistics(self) -> Dict[str, Any]: # Public method
        """Get current text correction performance statistics."""
        health = await self.health_check()
        return {
            **self.correction_stats,
            "agent_health_status": health,
            "current_config": self.get_config_summary_params()
        }
