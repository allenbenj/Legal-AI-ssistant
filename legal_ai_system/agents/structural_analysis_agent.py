# legal_ai_system/agents/structural_analysis/structural_analysis_agent.py
"""
StructuralAnalysisAgent - IRAC component extraction and document structure analysis.

Extracts and analyzes the structural components of legal documents using the IRAC framework
(Issue, Rule, Application, Conclusion) with enhanced legal document structure recognition.
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
class StructuralAnalysisResult:
    """Results from structural analysis of legal document."""

    irac_components: Dict[str, Any] = field(default_factory=dict)
    document_structure: Dict[str, Any] = field(default_factory=dict)
    section_analysis: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time_sec: float = 0.0  # Renamed for clarity
    model_used: str = ""
    structure_type: str = "unknown"  # e.g., brief, motion, opinion
    errors: List[str] = field(default_factory=list)
    analyzed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )  # Added

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Ensure IRAC components are initialized if not present after asdict
        if "irac_components" not in data or not data["irac_components"]:
            data["irac_components"] = {
                "issues": [],
                "rules": [],
                "application": [],
                "conclusion": [],
            }
        else:
            for key in ["issues", "rules", "application", "conclusion"]:
                if key not in data["irac_components"]:
                    data["irac_components"][key] = (
                        [] if key in ["issues", "rules"] else ""
                    )
        return data


class StructuralAnalysisAgent(BaseAgent, MemoryMixin):
    """
    Legal document structural analysis agent using IRAC framework.
    """

    def __init__(
        self, service_container: Any, **config: Any
    ):  # Renamed services, added config
        super().__init__(
            service_container, name="StructuralAnalysisAgent", agent_type="analysis"
        )

        # Get optimized Grok-Mini configuration for this agent
        self.llm_config = self.get_optimized_llm_config()
        self.logger.info(
            f"StructuralAnalysisAgent configured with model: {self.llm_config.get('llm_model', 'default')}"
        )
        self.llm_manager: Optional[LLMManager] = self.get_llm_manager()
        self.model_switcher: Optional[ModelSwitcher] = self._get_service(
            "model_switcher"
        )

        if not self.llm_manager:
            self.logger.error(
                "LLMManager service not available. StructuralAnalysisAgent may not function correctly."
            )
        # ModelSwitcher is optional, can proceed without it using default models

        # Configuration
        self.min_confidence_threshold = float(
            config.get("min_confidence_threshold", 0.7)
        )
        self.max_sections_per_analysis = int(
            config.get("max_sections_per_analysis", 20)
        )
        # self.section_analysis_enabled = bool(config.get('section_analysis_enabled', True)) # Used in prompt
        # self.detect_headers = bool(config.get('detect_headers', True)) # Used in prompt

        # IRAC structural analysis prompt template (remains largely the same)
        self.structural_prompt_template = """Extract IRAC (Issue, Rule, Application, Conclusion) components and analyze document structure.
IRAC FRAMEWORK SCHEMA:
{irac_schema}
DOCUMENT STRUCTURE ANALYSIS: ...
DOCUMENT TO ANALYZE:
{document_content}
ENTITY CONTEXT:
{entities_context}
SEMANTIC SUMMARY:
{semantic_summary}
ANALYSIS REQUIREMENTS:
1. IRAC COMPONENT EXTRACTION: ...
2. DOCUMENT STRUCTURE ANALYSIS: ...
3. SECTION-BY-SECTION ANALYSIS: ...
Return analysis in structured JSON format:
{{
    "irac_components": {{ "issues": [{{...}}], "rules": [{{...}}], "application": [{{...}}], "conclusion": [{{...}}] }},
    "document_structure": {{...}},
    "section_analysis": [{{...}}],
    "overall_confidence": 0.82,
    "analysis_notes": "..."
}}
Ensure precise identification of IRAC components with confidence ≥{min_confidence}. Focus on clear structural boundaries and legal reasoning flow."""

        # Performance tracking
        self.analysis_stats = {
            "total_analyses": 0,
            "avg_confidence": 0.0,
            "avg_irac_completeness_score": 0.0,  # Renamed
            "avg_sections_identified": 0.0,
            "processing_time_avg_sec": 0.0,
            "document_types_seen": {},
        }
        self.logger.info(
            "StructuralAnalysisAgent initialized.",
            parameters=self.get_config_summary_params(),
        )

    def get_config_summary_params(self) -> Dict[str, Any]:
        return {
            "min_conf": self.min_confidence_threshold,
            "max_sections": self.max_sections_per_analysis,
        }

    async def _process_task(
        self, task_data: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main processing logic.
        task_data: document_content, entities_context, semantic_summary, document_metadata
        """
        self.logger.info(
            "Starting structural analysis task.",
            parameters={"doc_id": metadata.get("document_id", "unknown")},
        )
        start_time_obj = datetime.now(timezone.utc)

        document_content = task_data.get("document_content", "")
        entities_context = task_data.get("entities_context", [])
        semantic_summary = task_data.get("semantic_summary", "")
        # document_metadata_param = task_data.get('document_metadata', {})

        if not document_content:
            self.logger.error("No document content provided for structural analysis.")
            return StructuralAnalysisResult(
                errors=["No document content provided."]
            ).to_dict()
        if not self.llm_manager:
            self.logger.error(
                "LLMManager not available, cannot perform structural analysis."
            )
            return StructuralAnalysisResult(
                errors=["LLMManager not available."]
            ).to_dict()

        try:
            complexity = self._assess_structural_complexity(document_content)
            model_to_use = self.llm_manager.primary_config.model
            provider_to_use = self.llm_manager.primary_config.provider.value

            if self.model_switcher:
                suggested_model_name = self.model_switcher.suggest_model_for_task(
                    "structural_analysis", complexity
                )
                if suggested_model_name:
                    model_to_use = suggested_model_name

            self.logger.info(
                f"Structural analysis with model.",
                parameters={
                    "model": model_to_use,
                    "provider": provider_to_use,
                    "complexity": complexity.value,
                },
            )

            irac_schema = self._build_irac_schema()
            entities_json = (
                json.dumps(entities_context[:10], indent=2)
                if entities_context
                else "None available"
            )

            prompt = self.structural_prompt_template.format(
                irac_schema=irac_schema,
                document_content=self._trim_content(document_content, 5000),
                entities_context=entities_json,
                semantic_summary=self._trim_content(semantic_summary, 1000),
                min_confidence=self.min_confidence_threshold,
            )

            llm_response_obj = await self.llm_manager.complete(
                prompt=prompt,
                model=model_to_use,
                provider=LLMProviderEnum(provider_to_use),
                temperature=0.1,
                max_tokens=4000,
            )

            analysis_data = self._parse_structural_response(
                llm_response_obj.content, document_content
            )  # Pass original for fallback

            confidence_score = analysis_data.get("overall_confidence", 0.0)
            structure_type = analysis_data.get("document_structure", {}).get(
                "document_type", "unknown"
            )
            processing_time_sec = (
                datetime.now(timezone.utc) - start_time_obj
            ).total_seconds()

            result = StructuralAnalysisResult(
                irac_components=analysis_data.get("irac_components", {}),
                document_structure=analysis_data.get("document_structure", {}),
                section_analysis=analysis_data.get("section_analysis", []),
                confidence_score=confidence_score,
                processing_time_sec=processing_time_sec,
                model_used=model_to_use,  # Or llm_response_obj.model_name
                structure_type=structure_type,
            )

            self._update_internal_analysis_stats(result)  # Renamed
            self.logger.info(
                "Structural analysis task completed.",
                parameters={
                    "doc_id": metadata.get("document_id", "unknown"),
                    "confidence": confidence_score,
                    "issues": len(result.irac_components.get("issues", [])),
                },
            )
            return result.to_dict()

        except LLMProviderError as e:
            self.logger.error(
                "LLMProviderError during structural analysis.", exception=e
            )
            return StructuralAnalysisResult(
                errors=[f"LLM Error: {str(e)}"], model_used=model_to_use
            ).to_dict()
        except Exception as e:
            self.logger.error(
                "Unexpected error during structural analysis.", exception=e
            )
            return StructuralAnalysisResult(
                errors=[f"Unexpected error: {str(e)}"], model_used=model_to_use
            ).to_dict()

    def _build_irac_schema(self) -> str:
        """Build IRAC schema with ontology guidance."""
        # ... (logic remains similar, ensure LegalEntityType is accessible)
        irac_concepts = {
            "LEGAL_ISSUE": "Legal questions or matters requiring determination",
            "RULE": "Legal principles, statutes, regulations, or precedents",
            "APPLICATION": "How legal rules apply to specific facts and circumstances",
            "CONCLUSION": "Legal outcomes, decisions, holdings, or recommendations",
        }
        schema_lines = []
        if LegalEntityType:  # Check if ontology was loaded
            for component_name_str, default_desc in irac_concepts.items():
                try:
                    entity_type_enum_member = getattr(
                        LegalEntityType, component_name_str, None
                    )
                    hint = (
                        entity_type_enum_member.value.prompt_hint
                        if entity_type_enum_member
                        else default_desc
                    )
                    schema_lines.append(f"- {component_name_str}: {hint}")
                except AttributeError:
                    schema_lines.append(
                        f"- {component_name_str}: {default_desc}"
                    )  # Fallback
        else:
            self.logger.warning(
                "LegalEntityType ontology not available for building IRAC schema."
            )
            schema_lines = [f"- {name}: {desc}" for name, desc in irac_concepts.items()]
        return "\n".join(schema_lines)

    def _parse_structural_response(
        self, response_content: str, original_doc_content: str
    ) -> Dict[str, Any]:
        """Parse LLM response into structured analysis data."""
        # ... (logic remains similar, ensure robust JSON parsing)
        try:
            json_content = response_content
            if "```json" in response_content:
                json_content = response_content.split("```json")[1].split("```")[0]
            elif (
                "```" in response_content
                and response_content.strip().startswith("```")
                and response_content.strip().endswith("```")
            ):
                json_content = response_content.strip()[3:-3]

            parsed_data = json.loads(json_content.strip())

            # Basic validation and normalization
            validated_data = {
                "irac_components": self._validate_irac_components(
                    parsed_data.get("irac_components", {})
                ),
                "document_structure": self._validate_document_structure(
                    parsed_data.get("document_structure", {})
                ),
                "section_analysis": self._validate_section_analysis(
                    parsed_data.get("section_analysis", [])
                ),
                "overall_confidence": float(parsed_data.get("overall_confidence", 0.0)),
                "analysis_notes": parsed_data.get("analysis_notes", ""),
            }
            return validated_data
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.logger.warning(
                f"Failed to parse LLM structural response. Content: {response_content[:200]}...",
                exception=e,
            )
            return {  # Return default structure on error
                "irac_components": {
                    "issues": [],
                    "rules": [],
                    "application": "",
                    "conclusion": "",
                },
                "document_structure": {},
                "section_analysis": [],
                "overall_confidence": 0.0,
                "analysis_notes": f"Response parsing error: {str(e)}",
            }

    def _validate_list_of_dicts(self, data_list: Any) -> List[Dict[str, Any]]:
        """Helper to ensure a list contains dictionaries."""
        if not isinstance(data_list, list):
            return []
        return [item for item in data_list if isinstance(item, dict)]

    def _validate_irac_components(self, irac_data: Any) -> Dict[str, Any]:
        """Validate and normalize IRAC component data."""
        if not isinstance(irac_data, dict):
            return {"issues": [], "rules": [], "application": "", "conclusion": ""}

        validated_irac: Dict[str, Any] = {
            "issues": [],
            "rules": [],
            "application": [],
            "conclusion": [],
        }
        components_list_keys = [
            "issues",
            "rules",
            "application",
            "conclusion",
        ]  # Application/Conclusion also lists of dicts per prompt

        for component_key in components_list_keys:
            items = irac_data.get(component_key, [])
            if not isinstance(items, list):
                items = (
                    [str(items)] if items else []
                )  # Handle if LLM returns string for app/conclusion

            validated_items = []
            for item_data in items:
                if (
                    isinstance(item_data, dict)
                    and "text" in item_data
                    and "confidence" in item_data
                ):
                    if float(item_data["confidence"]) >= self.min_confidence_threshold:
                        validated_items.append(item_data)
                elif isinstance(item_data, str) and component_key in [
                    "application",
                    "conclusion",
                ]:
                    validated_items.append(
                        {"text": item_data, "confidence": self.min_confidence_threshold}
                    )

            validated_irac[component_key] = validated_items
        return validated_irac

    def _validate_document_structure(self, structure_data: Any) -> Dict[str, Any]:
        """Validate and normalize document_structure returned by the LLM."""
        if not isinstance(structure_data, dict):
            return {}

        validated: Dict[str, Any] = {}

        doc_type = structure_data.get("document_type")
        if isinstance(doc_type, str):
            validated["document_type"] = doc_type

        sections = structure_data.get("sections")
        if isinstance(sections, list):
            validated["sections"] = [s for s in sections if isinstance(s, dict)]

        # Preserve other simple key/value pairs
        for key, value in structure_data.items():
            if key in ("document_type", "sections"):
                continue
            if isinstance(value, (str, int, float, bool, dict, list)):
                validated[key] = value

        return validated

    def _validate_section_analysis(self, sections_data: Any) -> List[Dict[str, Any]]:
        """Validate section_analysis ensuring consistent structure."""
        if not isinstance(sections_data, list):
            return []

        validated_sections: List[Dict[str, Any]] = []
        for entry in sections_data[: self.max_sections_per_analysis]:
            if not isinstance(entry, dict):
                continue

            cleaned_entry: Dict[str, Any] = {}

            title = entry.get("section_title") or entry.get("title")
            if isinstance(title, str):
                cleaned_entry["section_title"] = title

            analysis_text = entry.get("analysis") or entry.get("summary")
            if isinstance(analysis_text, str):
                cleaned_entry["analysis"] = analysis_text

            confidence = entry.get("confidence")
            if confidence is not None:
                try:
                    conf_value = float(confidence)
                    if conf_value >= self.min_confidence_threshold:
                        cleaned_entry["confidence"] = conf_value
                except (TypeError, ValueError):
                    pass

            for key in ("start_index", "end_index", "section_text", "irac"):
                if key in entry and isinstance(
                    entry[key], (str, int, float, dict, list)
                ):
                    cleaned_entry[key] = entry[key]

            if cleaned_entry:
                validated_sections.append(cleaned_entry)

        return validated_sections

    def _assess_structural_complexity(self, document_content: str) -> TaskComplexity:
        """Assess structural analysis complexity for model selection."""
        # ... (logic remains similar)
        content_length = len(document_content)
        if content_length < 1500:
            complexity = TaskComplexity.SIMPLE
        elif content_length > 6000:
            complexity = TaskComplexity.COMPLEX
        else:
            complexity = TaskComplexity.MODERATE
        return complexity

    def _trim_content(
        self, content: str, max_length_chars: int
    ) -> str:  # Clarified unit
        """Trim content to maximum length with ellipsis."""
        if len(content) <= max_length_chars:
            return content
        return content[: max_length_chars - len("... [TRUNCATED]")] + "... [TRUNCATED]"

    def _update_internal_analysis_stats(
        self, result: StructuralAnalysisResult
    ):  # Renamed
        """Update internal performance statistics for this agent."""
        # ... (logic largely remains, ensure keys match StructuralAnalysisResult and self.analysis_stats)
        self.analysis_stats["total_analyses"] += 1
        total = self.analysis_stats["total_analyses"]
        self.analysis_stats["avg_confidence"] = (
            self.analysis_stats["avg_confidence"] * (total - 1)
            + result.confidence_score
        ) / total

        # IRAC completeness (simple score based on presence of components)
        irac = result.irac_components
        completeness_score = 0.0
        if irac.get("issues"):
            completeness_score += 0.25
        if irac.get("rules"):
            completeness_score += 0.25
        if irac.get("application"):
            completeness_score += 0.25
        if irac.get("conclusion"):
            completeness_score += 0.25
        self.analysis_stats["avg_irac_completeness_score"] = (
            self.analysis_stats["avg_irac_completeness_score"] * (total - 1)
            + completeness_score
        ) / total

        self.analysis_stats["avg_sections_identified"] = (
            self.analysis_stats["avg_sections_identified"] * (total - 1)
            + len(result.section_analysis)
        ) / total
        self.analysis_stats["processing_time_avg_sec"] = (
            self.analysis_stats["processing_time_avg_sec"] * (total - 1)
            + result.processing_time_sec
        ) / total

        doc_type = result.structure_type
        self.analysis_stats["document_types_seen"][doc_type] = (
            self.analysis_stats["document_types_seen"].get(doc_type, 0) + 1
        )

    async def get_analysis_statistics(self) -> Dict[str, Any]:  # Public method
        """Get current structural analysis performance statistics."""
        # ... (logic remains similar)
        health = await self.health_check()
        return {
            **self.analysis_stats,
            "agent_health_status": health,
            "current_config": self.get_config_summary_params(),
        }
