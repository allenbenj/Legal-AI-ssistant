# legal_ai_system/agents/violation_detector/violation_detector_agent.py
"""
Violation Detector Agent - Specialized Legal Violation Detection
This agent identifies and analyzes various types of legal violations.
"""

import json
import re
import sys
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for absolute imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from legal_ai_system.config.agent_unified_config import create_agent_memory_mixin
    from legal_ai_system.core.base_agent import BaseAgent, ProcessingResult
    from legal_ai_system.core.detailed_logging import LogCategory
    from legal_ai_system.core.llm_providers import (
        LLMManager,
        LLMProviderEnum,
        LLMProviderError,
    )
    from legal_ai_system.core.unified_exceptions import (  # Fixed class name
        AgentExecutionError,
        AgentProcessingError,
    )
    from legal_ai_system.memory.unified_memory_manager import MemoryType
except ImportError:
    # Fallback for relative imports
    try:
        from ..core.agent_unified_config import create_agent_memory_mixin
        from ..core.base_agent import BaseAgent, ProcessingResult
        from ..core.detailed_logging import LogCategory
        from ..core.llm_providers import LLMManager, LLMProviderEnum, LLMProviderError
        from ..core.unified_exceptions import (  # Fixed class name
            AgentExecutionError,
            AgentProcessingError,
        )
        from ..core.unified_memory_manager import MemoryType
    except ImportError:
        # Final fallback - create minimal classes
        class BaseAgent:
            def __init__(self, service_container, name="Agent", agent_type="base"):
                self.service_container = service_container
                self.name = name
                self.agent_type = agent_type
                import logging

                self.logger = logging.getLogger(name)

            def _get_service(self, name):
                return getattr(self.service_container, "get_service", lambda x: None)(
                    name
                )

            def get_optimized_llm_config(self):
                return {}

        class ProcessingResult:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        class LLMManager:
            pass

        class LLMProviderError(Exception):
            pass

        class LLMProviderEnum:
            XAI = "xai"

        class AgentExecutionError(Exception):
            pass

        class AgentProcessingError(Exception):
            pass

        class LogCategory:
            AGENT = "AGENT"

        class MemoryType:
            AGENT_SPECIFIC = "agent_specific"

        def create_agent_memory_mixin():
            class MemoryMixin:
                pass

            return MemoryMixin


# Create memory mixin for agents
MemoryMixin = create_agent_memory_mixin()


@dataclass
class DetectedViolation:
    violation_type: str
    description: str
    context: str  # Surrounding text for context
    confidence: float
    severity: str  # e.g., "critical", "high", "medium", "low"
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    detected_by: str = "unknown"  # "pattern_matching", "llm_analysis"
    reasoning: Optional[str] = None
    relevant_entities_ids: List[str] = field(default_factory=list)
    supporting_precedents: List[str] = field(default_factory=list)
    violation_id: str = field(
        default_factory=lambda: f"VIOL_{uuid.uuid4().hex[:8]}"
    )  # Unique ID

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ViolationDetectionOutput:
    document_id: str
    violations_found: List[DetectedViolation] = field(default_factory=list)
    analysis_summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    overall_confidence: float = 0.0
    processing_time_sec: float = 0.0
    errors: List[str] = field(default_factory=list)
    model_used_for_llm: Optional[str] = None  # Track LLM model if used

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["violations_found"] = [v.to_dict() for v in self.violations_found]
        return data


class ViolationDetectorAgent(BaseAgent):
    """
    Specialized agent for detecting legal violations in documents and case materials.
    Combines pattern matching and LLM-based analysis.
    """

    def __init__(self, service_container: Any, **config: Any):
        super().__init__(
            service_container,
            name="ViolationDetectorAgent",
            agent_type="legal_analysis",
        )

        # Get optimized Grok-Mini configuration for this agent
        self.llm_config = self.get_optimized_llm_config()
        self.logger.info(
            f"ViolationDetectorAgentAgent configured with model: {self.llm_config.get('llm_model', 'default')}"
        )
        self.llm_manager: Optional[LLMManager] = self._get_service(
            "llm_manager"
        )  # Use helper

        self.config = config
        self.min_pattern_confidence = float(config.get("min_pattern_confidence", 0.55))
        self.min_llm_confidence = float(config.get("min_llm_confidence", 0.65))
        self.enable_llm_analysis = bool(
            config.get("enable_llm_analysis", True)
        )  # Renamed from enable_llm_validation
        self.max_text_for_llm = int(config.get("max_text_for_llm", 8000))  # Chars

        self._init_violation_patterns()

        self.logger.info(
            f"{self.name} initialized.",
            parameters={
                "llm_analysis_enabled": self.enable_llm_analysis,
                "llm_manager_available": self.llm_manager is not None,
            },
        )

    def _init_violation_patterns(self):
        """Initialize violation detection patterns and keywords from config or defaults."""
        # Default patterns (can be overridden by config)
        default_patterns = {
            "Brady Violation": [
                r"brady\s+violation",
                r"exculpatory\s+evidence\s+(?:withheld|not\s+disclosed|suppressed)",
                r"failure\s+to\s+disclose\s+favorable\s+evidence",
            ],
            "4th Amendment Violation": [
                r"unreasonable\s+search(?:es)?\s+and\s+seizure(?:s)?",
                r"warrantless\s+search",
                r"illegal\s+seizure",
                r"lack\s+of\s+probable\s+cause",
            ],
            "5th Amendment Violation": [
                r"miranda\s+rights?\s+(?:violation|not\s+read)",
                r"self[- ]incrimination",
                r"coerced\s+confession",
                r"right\s+to\s+remain\s+silent\s+violated",
            ],
            "Law Enforcement Misconduct": [
                r"police\s+brutality",
                r"excessive\s+force",
                r"false\s+arrest",
                r"fabricated\s+evidence\s+by\s+police",
                r"perjury\s+by\s+officer",
            ],
            "Evidence Tampering": [
                r"evidence\s+tampering",
                r"chain\s+of\s+custody\s+(?:broken|violated|compromised)",
                r"altered\s+evidence",
                r"planted\s+evidence",
            ],
            "Witness Tampering": [
                r"witness\s+intimidation",
                r"witness\s+tampering",
                r"coaching\s+witness\s+improperly",
                r"threatening\s+a\s+witness",
            ],
        }
        # Load from config if available, else use defaults
        self.violation_patterns_config = self.config.get(
            "violation_patterns", default_patterns
        )

        # Create a flat list for iteration
        self.all_violation_pattern_groups: List[Tuple[str, List[str]]] = []
        for viol_type, patterns in self.violation_patterns_config.items():
            if isinstance(patterns, list) and all(isinstance(p, str) for p in patterns):
                self.all_violation_pattern_groups.append((viol_type, patterns))
            else:
                self.logger.warning(
                    f"Invalid pattern format for violation type '{viol_type}' in config. Skipping."
                )
        self.logger.debug(
            f"Initialized {len(self.all_violation_pattern_groups)} violation pattern groups."
        )

    async def _process_task(
        self, task_data: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        document_id = metadata.get("document_id", f"unknown_doc_{uuid.uuid4().hex[:8]}")
        text_content = task_data.get("text", "")

        self.logger.info(
            f"Starting violation detection for doc '{document_id}'.",
            parameters={"text_len": len(text_content)},
        )
        start_time_obj = datetime.now(timezone.utc)

        output = ViolationDetectionOutput(document_id=document_id)

        if not text_content or len(text_content.strip()) < 50:
            self.logger.warning(
                f"Insufficient text content for violation detection (doc '{document_id}')."
            )
            output.errors.append("Insufficient text content provided.")
            output.processing_time_sec = (
                datetime.now(timezone.utc) - start_time_obj
            ).total_seconds()
            return output.to_dict()

        try:
            (
                detected_violations_list,
                model_used,
            ) = await self._detect_all_violations_async(text_content, document_id)
            output.violations_found = self._filter_and_finalize_violations(
                detected_violations_list
            )
            output.model_used_for_llm = model_used

            if output.violations_found:
                output.analysis_summary = await self._analyze_detected_violations(
                    output.violations_found, text_content
                )
                output.recommendations = (
                    await self._generate_recommendations_for_violations(
                        output.violations_found, output.analysis_summary
                    )
                )

            if output.violations_found:
                output.overall_confidence = round(
                    sum(v.confidence for v in output.violations_found)
                    / len(output.violations_found),
                    3,
                )
            else:  # No violations found, high confidence in that assessment (assuming thorough check)
                output.overall_confidence = 0.95

            self.logger.info(
                f"Violation detection completed for doc '{document_id}'.",
                parameters={
                    "violations_found": len(output.violations_found),
                    "overall_conf": output.overall_confidence,
                },
            )

        except AgentProcessingError as ape:  # Catch specific agent errors
            self.logger.error(
                f"AgentProcessingError during violation detection for doc '{document_id}'.",
                exception=ape,
                exc_info=True,
            )
            output.errors.append(f"Agent processing error: {str(ape)}")
            output.overall_confidence = 0.1  # Low confidence due to error
        except Exception as e:  # Catch unexpected errors
            self.logger.error(
                f"Unexpected error during violation detection for doc '{document_id}'.",
                exception=e,
                exc_info=True,
            )
            output.errors.append(f"Unexpected error: {str(e)}")
            output.overall_confidence = 0.1

        finally:
            output.processing_time_sec = round(
                (datetime.now(timezone.utc) - start_time_obj).total_seconds(), 3
            )

        return output.to_dict()

    async def _detect_all_violations_async(
        self, text: str, doc_id: str
    ) -> Tuple[List[DetectedViolation], Optional[str]]:
        self.logger.debug(f"Detecting all violations for doc '{doc_id}'.")
        violations: List[DetectedViolation] = []
        model_used: Optional[str] = None

        # 1. Pattern-based detection
        for violation_type_label, regex_patterns in self.all_violation_pattern_groups:
            pattern_matches = self._find_matches_for_patterns(
                text, regex_patterns, violation_type_label
            )
            violations.extend(pattern_matches)

        self.logger.info(
            f"Pattern matching found {len(violations)} potential violations for doc '{doc_id}'."
        )

        # 2. LLM-based analysis (validation and new detection)
        if self.enable_llm_analysis and self.llm_manager:
            try:
                # Truncate text if too long for LLM
                text_for_llm = (
                    text
                    if len(text) <= self.max_text_for_llm
                    else text[: self.max_text_for_llm]
                )
                if len(text) > self.max_text_for_llm:
                    self.logger.warning(
                        f"Text for doc '{doc_id}' truncated to {self.max_text_for_llm} chars for LLM analysis."
                    )

                (
                    llm_detected_violations,
                    model_used,
                ) = await self._llm_analyze_text_for_violations(
                    text_for_llm, violations, doc_id
                )
                violations = self._merge_llm_and_pattern_violations(
                    violations, llm_detected_violations
                )
                self.logger.info(
                    f"LLM analysis resulted in {len(violations)} total potential violations for doc '{doc_id}'."
                )
            except Exception as llm_e:  # Catch errors specifically from the LLM step
                self.logger.error(
                    f"LLM violation analysis failed for doc '{doc_id}'. Pattern results will be used.",
                    exception=llm_e,
                )
                # `violations` will still contain pattern-based results.
        return violations, model_used

    def _find_matches_for_patterns(
        self, text: str, patterns_list: List[str], violation_type: str
    ) -> List[DetectedViolation]:
        matches_found: List[DetectedViolation] = []
        for regex_str_pattern in patterns_list:
            try:
                for match_obj in re.finditer(
                    regex_str_pattern, text, re.IGNORECASE | re.DOTALL
                ):
                    start_idx, end_idx = match_obj.span()
                    # Wider context for better understanding, snippet for description
                    context_text = text[
                        max(0, start_idx - 250) : min(len(text), end_idx + 250)
                    ].strip()
                    matched_text_snippet = match_obj.group(0).strip()

                    confidence_val = self._calculate_pattern_match_confidence(
                        regex_str_pattern, matched_text_snippet
                    )

                    if confidence_val >= self.min_pattern_confidence:
                        matches_found.append(
                            DetectedViolation(
                                violation_type=violation_type,
                                description=matched_text_snippet,
                                context=context_text,
                                confidence=confidence_val,
                                severity=self._assess_violation_severity(
                                    violation_type, matched_text_snippet
                                ),
                                start_pos=start_idx,
                                end_pos=end_idx,
                                detected_by="pattern_matching",
                                reasoning=f"Matched pattern: '{regex_str_pattern}'",
                            )
                        )
            except re.error as re_err:
                self.logger.warning(
                    f"Regex error during violation pattern matching for type '{violation_type}'.",
                    parameters={"pattern": regex_str_pattern, "error": str(re_err)},
                )
        return matches_found

    def _calculate_pattern_match_confidence(
        self, pattern_str: str, matched_text: str
    ) -> float:
        base_conf = 0.55
        if len(pattern_str) > 20:
            base_conf += 0.1  # Longer patterns are more specific
        if len(matched_text.split()) > 3:
            base_conf += 0.1  # Multi-word matches are better

        critical_terms = [
            "brady",
            "miranda",
            "unconstitutional",
            "tampering",
            "perjury",
            "fabricated",
            "coerced",
        ]
        if any(term in matched_text.lower() for term in critical_terms):
            base_conf += 0.2

        return round(min(0.95, base_conf), 3)

    def _assess_violation_severity(
        self, violation_type_label: str, description_text: str
    ) -> str:
        desc_lower = description_text.lower()
        type_lower = violation_type_label.lower()

        if any(
            kw in desc_lower
            for kw in [
                "fabricated evidence",
                "planted evidence",
                "destroyed evidence",
                "coerced confession",
                "perjury by officer",
                "witness intimidation by force",
            ]
        ):
            return "critical"
        if (
            "brady" in type_lower
            or "constitutional" in type_lower
            or "4th amendment" in type_lower
            or "5th amendment" in type_lower
            or any(
                kw in desc_lower
                for kw in [
                    "suppressed evidence",
                    "evidence withheld",
                    "false testimony",
                ]
            )
        ):
            return "high"
        if (
            "misconduct" in type_lower
            or "tampering" in type_lower
            or any(
                kw in desc_lower
                for kw in ["excessive force", "false arrest", "improper influence"]
            )
        ):
            return "medium"
        return "low"

    async def _llm_analyze_text_for_violations(
        self,
        text_for_llm: str,
        current_violations: List[DetectedViolation],
        doc_id: str,
    ) -> Tuple[List[DetectedViolation], Optional[str]]:
        if not self.llm_manager:
            return [], None
        self.logger.debug(
            f"Using LLM for violation analysis and discovery on doc '{doc_id}'."
        )

        # Summarize existing pattern-based findings for the LLM prompt
        pattern_findings_summary = (
            "Potential Violations (from patterns, to be validated by you):\n"
        )
        if current_violations:
            for i, v in enumerate(current_violations[:5]):  # Limit summary size
                pattern_findings_summary += f"{i+1}. Type: {v.violation_type}, Snippet: '{v.description[:100]}...', Confidence: {v.confidence:.2f}\n"
        else:
            pattern_findings_summary = (
                "No violations found by initial pattern matching."
            )

        prompt = f"""
        You are an expert legal AI tasked with identifying potential legal violations in a document.
        Analyze the provided DOCUMENT TEXT EXCERPT. Focus on identifying instances of:
        - Brady Violations (withholding exculpatory evidence)
        - 4th Amendment Violations (unreasonable search/seizure)
        - 5th Amendment Violations (self-incrimination, Miranda rights)
        - Law Enforcement Misconduct (excessive force, false arrest, perjury)
        - Evidence Tampering (altering, destroying, planting evidence, chain of custody issues)
        - Witness Tampering/Intimidation

        {pattern_findings_summary}

        DOCUMENT TEXT EXCERPT:
        ---
        {text_for_llm}
        ---
        
        INSTRUCTIONS:
        1. Review the 'Potential Violations' found by patterns. For each, decide if it's a valid violation based on the full DOCUMENT TEXT EXCERPT.
           If valid, refine its details. If not, discard it.
        2. Identify any ADDITIONAL violations in the DOCUMENT TEXT EXCERPT not listed above.
        3. For ALL confirmed or newly identified violations, provide the following in a JSON array format:
           - "violation_id": (string, a unique ID you generate for this finding, e.g., "LLM_VIOL_1")
           - "violation_type": (string, one of the focused types above, or a more specific legal term if clear)
           - "description": (string, a concise summary of the violation)
           - "exact_quote": (string, the exact text snippet from the document that indicates the violation - crucial for grounding)
           - "confidence": (float, 0.0-1.0, your confidence that this is a genuine violation based on the text)
           - "severity": (string, "low", "medium", "high", or "critical")
           - "reasoning": (string, brief explanation for why this constitutes a violation and your confidence level)
           - "supporting_precedents": (list of strings, optional, e.g., ["Brady v. Maryland"])
        4. If no violations are found or confirmed, return an empty JSON array [].
        5. Only include violations where your confidence is >= {self.min_llm_confidence}.
        
        STRICT JSON OUTPUT FORMAT (an array of objects):
        """
        model_to_use = self.config.get(
            "llm_model_for_violation_detection", self.llm_manager.primary_config.model
        )
        provider_to_use = self.config.get(
            "llm_provider_for_violation_detection",
            self.llm_manager.primary_config.provider.value,
        )

        self.logger.debug(
            f"Sending violation analysis prompt to LLM ({provider_to_use}/{model_to_use}) for doc '{doc_id}'."
        )

        try:
            # Assuming LLMManager has a method that takes model and provider, or uses primary if not specified.
            llm_response = await self.llm_manager.complete(
                prompt,
                model=model_to_use,
                provider=LLMProviderEnum(provider_to_use),
                model_params={"temperature": 0.1, "max_tokens": 2000},
            )
            model_used = f"{provider_to_use}/{model_to_use}"
            parsed = self._parse_llm_violation_response(
                llm_response.content, text_for_llm, doc_id
            )
            return parsed, model_used
        except LLMProviderError as e:
            self.logger.error(
                f"LLM API call for violation analysis failed for doc '{doc_id}'.",
                exception=e,
            )
            raise AgentProcessingError(
                "LLM analysis failed.", underlying_exception=e
            ) from e
        except Exception as e:
            self.logger.error(
                f"Unexpected error in LLM violation analysis for doc '{doc_id}'.",
                exception=e,
            )
            raise AgentProcessingError(
                "Unexpected error during LLM processing.", underlying_exception=e
            ) from e

    def _parse_llm_violation_response(
        self, response_content: str, original_text: str, doc_id: str
    ) -> List[DetectedViolation]:
        llm_violations: List[DetectedViolation] = []
        try:
            # Robust JSON extraction from LLM response
            json_match = re.search(
                r"\[\s*(\{[\s\S]*?\})\s*\]", response_content, re.DOTALL
            )  # Try to find array of objects
            if not json_match:
                json_match_single = re.search(
                    r"(\{[\s\S]*?\})", response_content, re.DOTALL
                )  # Try to find single object if array fails
                if json_match_single:
                    json_str = (
                        f"[{json_match_single.group(1)}]"  # Wrap single object in array
                    )
                else:
                    self.logger.warning(
                        f"No JSON array or object found in LLM violation response for doc '{doc_id}'. Response start: {response_content[:150]}"
                    )
                    return []
            else:
                json_str = json_match.group(0)  # The full array string

            violations_data_list = json.loads(json_str)
            if not isinstance(
                violations_data_list, list
            ):  # Ensure it's a list after parsing
                self.logger.warning(
                    f"Parsed LLM violation response is not a list for doc '{doc_id}'. Found: {type(violations_data_list)}"
                )
                return []

            for item_data in violations_data_list:
                if not isinstance(item_data, dict):
                    continue

                confidence_val = float(item_data.get("confidence", 0.0))
                if confidence_val < self.min_llm_confidence:
                    continue

                description = item_data.get("description", "")
                exact_quote = item_data.get("exact_quote", "")

                # Attempt to find the span of the exact_quote in the original_text
                start_pos, end_pos = None, None
                if exact_quote:
                    try:
                        match_quote = re.search(
                            re.escape(exact_quote[:100]), original_text
                        )  # Match first 100 chars for robustness
                        if match_quote:
                            start_pos, end_pos = match_quote.span()
                    except re.error:
                        self.logger.warning(
                            f"Regex error finding exact quote for LLM violation on doc '{doc_id}'. Quote: {exact_quote[:50]}"
                        )

                violation = DetectedViolation(
                    violation_id=item_data.get(
                        "violation_id", self._generate_unique_id("LLM_VIOL")
                    ),
                    violation_type=item_data.get(
                        "violation_type", "Unknown LLM Violation"
                    ),
                    description=description
                    if description
                    else exact_quote,  # Prefer description, fallback to quote
                    context=exact_quote
                    if exact_quote
                    else description[:250],  # Use quote as context if available
                    confidence=confidence_val,
                    severity=str(item_data.get("severity", "medium")).lower(),
                    start_pos=start_pos,
                    end_pos=end_pos,
                    detected_by="llm_analysis",
                    reasoning=item_data.get("reasoning"),
                    supporting_precedents=item_data.get("supporting_precedents", []),
                )
                llm_violations.append(violation)
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to parse JSON from LLM violation response for doc '{doc_id}'. JSON string was: '{json_str if 'json_str' in locals() else response_content[:200]}'",
                exception=e,
            )
        except Exception as e:  # Catch other errors during parsing individual items
            self.logger.error(
                f"Unexpected error parsing an LLM violation item for doc '{doc_id}'. Item data: {item_data if 'item_data' in locals() else 'N/A'}",
                exception=e,
            )

        self.logger.debug(
            f"LLM response parsing yielded {len(llm_violations)} violations for doc '{doc_id}'."
        )
        return llm_violations

    def _merge_llm_and_pattern_violations(
        self,
        pattern_violations: List[DetectedViolation],
        llm_violations: List[DetectedViolation],
    ) -> List[DetectedViolation]:
        """Merges violations, prioritizing LLM insights or higher confidence for overlaps."""
        # If LLM provides a comprehensive list meant to replace/validate patterns,
        # it might be simpler to just use LLM results if they are deemed superior.
        # For now, a more nuanced merge:
        if not llm_violations:
            return pattern_violations  # No LLM results, keep pattern ones
        if not pattern_violations:
            return llm_violations  # No pattern results, use LLM ones

        final_violations: List[DetectedViolation] = []

        # Index LLM violations by a robust key (e.g., type + normalized start of description)
        # for quick lookup. LLM violations are considered "more authoritative" if they overlap.
        llm_map: Dict[Tuple[str, str], DetectedViolation] = {}
        for lv in llm_violations:
            key_desc = re.sub(
                r"\s+", " ", lv.description.lower().strip()[:60]
            )  # Normalized first 60 chars
            llm_map[(lv.violation_type.lower(), key_desc)] = lv
            final_violations.append(lv)  # Add all LLM violations first

        # Add pattern violations only if a similar one (type + description snippet) isn't already added by LLM
        for pv in pattern_violations:
            key_desc_pv = re.sub(r"\s+", " ", pv.description.lower().strip()[:60])
            key_pv = (pv.violation_type.lower(), key_desc_pv)

            found_matching_llm = False
            for (llm_type_key, llm_desc_key), llm_viol_obj in llm_map.items():
                # Check type match and if description snippets are very similar (e.g., Jaccard on words or overlap)
                # Simple check: if LLM key contains pattern key, or vice versa (for slight variations)
                if llm_type_key == key_pv[0] and (
                    key_pv[1] in llm_desc_key or llm_desc_key in key_pv[1]
                ):
                    found_matching_llm = True
                    # Potentially update the LLM violation with info from pattern one if pattern has better span/confidence
                    if (
                        pv.confidence > llm_viol_obj.confidence
                    ):  # Example: take higher confidence
                        llm_viol_obj.confidence = pv.confidence
                    if (
                        pv.start_pos is not None and llm_viol_obj.start_pos is None
                    ):  # Prefer pattern span if LLM lacks one
                        llm_viol_obj.start_pos = pv.start_pos
                        llm_viol_obj.end_pos = pv.end_pos
                    break

            if not found_matching_llm:
                final_violations.append(pv)

        # Deduplicate the final list more thoroughly if needed (e.g., span overlap)
        # For now, this simple merge favors LLM for overlaps.
        return final_violations

    def _filter_and_finalize_violations(
        self, violations: List[DetectedViolation]
    ) -> List[DetectedViolation]:
        """Final filtering, e.g. by confidence, and deduplication based on span for very close items."""
        if not violations:
            return []

        # 1. Filter by overall confidence threshold (applied again after merge)
        confident_violations = [
            v
            for v in violations
            if v.confidence >= self.min_llm_confidence
            or (
                v.detected_by == "pattern_matching"
                and v.confidence >= self.min_pattern_confidence
            )
        ]

        if not confident_violations:
            return []

        # 2. Deduplicate by span overlap if spans are available and very similar
        # Sort by confidence (desc) then start_pos (asc) to process better ones first
        confident_violations.sort(
            key=lambda v: (
                -v.confidence,
                v.start_pos if v.start_pos is not None else float("inf"),
            )
        )

        finalized_list: List[DetectedViolation] = []
        for viol in confident_violations:
            is_duplicate = False
            for existing_viol in finalized_list:
                # Check if types are same and spans overlap significantly
                if (
                    viol.violation_type == existing_viol.violation_type
                    and viol.start_pos is not None
                    and viol.end_pos is not None
                    and existing_viol.start_pos is not None
                    and existing_viol.end_pos is not None
                ):
                    # Simple overlap: if one contains the other or substantial overlap
                    overlap_start = max(viol.start_pos, existing_viol.start_pos)
                    overlap_end = min(viol.end_pos, existing_viol.end_pos)
                    if overlap_end > overlap_start:  # They overlap
                        # Define "significant overlap" (e.g., >70% of shorter violation's length)
                        overlap_len = overlap_end - overlap_start
                        min_len = min(
                            viol.end_pos - viol.start_pos,
                            existing_viol.end_pos - existing_viol.start_pos,
                        )
                        if min_len > 0 and (overlap_len / min_len) > 0.7:
                            is_duplicate = True
                            # The current sort already prioritizes higher confidence, so `existing_viol` is likely better or equal.
                            # We could merge details here if needed, but for now, just skip the duplicate.
                            self.logger.trace(
                                f"Deduplicating violation '{viol.violation_id}' due to overlap with '{existing_viol.violation_id}'."
                            )
                            break
            if not is_duplicate:
                finalized_list.append(viol)

        return finalized_list

    async def _analyze_detected_violations(
        self, violations: List[DetectedViolation], text_content: str
    ) -> Dict[str, Any]:
        if not violations:
            return {
                "total_violations": 0,
                "severity_summary": {},
                "potential_impacts": ["No violations detected."],
            }

        severity_summary = defaultdict(int)
        violation_types_summary = defaultdict(int)
        unique_precedents = set()

        for v in violations:
            severity_summary[v.severity] += 1
            violation_types_summary[v.violation_type] += 1
            if v.supporting_precedents:
                unique_precedents.update(v.supporting_precedents)

        potential_impacts = [
            "Review detected violations for specific case impact."
        ]  # Default
        critical_count = severity_summary.get("critical", 0)
        high_count = severity_summary.get("high", 0)

        if critical_count > 0:
            potential_impacts = [
                f"CRITICAL: {critical_count} violation(s) may warrant immediate action (e.g., motion to dismiss, sanctions)."
            ]
        elif high_count > 0:
            potential_impacts = [
                f"HIGH: {high_count} violation(s) suggest significant legal challenges, potential evidence suppression, or grounds for appeal."
            ]

        # Placeholder for future LLM-based impact summary:
        # impact_summary_llm = await self.llm_manager.summarize_impact(violations, text_content)

        return {
            "total_violations": len(violations),
            "severity_summary": dict(severity_summary),
            "violation_types_summary": dict(violation_types_summary),
            "potential_impacts": potential_impacts,
            "key_precedents_identified": sorted(list(unique_precedents)),
        }

    async def _generate_recommendations_for_violations(
        self, violations: List[DetectedViolation], analysis_summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        recommendations: List[Dict[str, Any]] = []
        if not violations:
            return recommendations

        critical_count = analysis_summary.get("severity_summary", {}).get("critical", 0)
        high_count = analysis_summary.get("severity_summary", {}).get("high", 0)

        if critical_count > 0:
            recommendations.append(
                {
                    "priority": "URGENT",
                    "action": "Immediate legal counsel review required. Consider motions for dismissal, sanctions, or mistrial.",
                    "reasoning": f"{critical_count} critical severity violation(s) detected.",
                }
            )
        elif high_count > 0:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "action": "Thorough investigation and strategic response required. Consider motions to suppress, further discovery, or appeal strategy.",
                    "reasoning": f"{high_count} high severity violation(s) detected.",
                }
            )

        unique_violation_types = analysis_summary.get(
            "violation_types_summary", {}
        ).keys()
        for v_type in unique_violation_types:
            if "Brady" in v_type:
                recommendations.append(
                    {
                        "priority": "HIGH",
                        "action": "Scrutinize all discovery for withheld exculpatory or impeachment evidence. File Brady motion if applicable.",
                        "reasoning": f"Potential {v_type}.",
                    }
                )
            elif "4th Amendment" in v_type:
                recommendations.append(
                    {
                        "priority": "HIGH",
                        "action": "Evaluate legality of search/seizure. Consider motion to suppress resulting evidence.",
                        "reasoning": f"Potential {v_type}.",
                    }
                )
            elif "5th Amendment" in v_type:
                recommendations.append(
                    {
                        "priority": "HIGH",
                        "action": "Assess circumstances of any statements/confessions for coercion or Miranda violations.",
                        "reasoning": f"Potential {v_type}.",
                    }
                )
            elif "Evidence Tampering" in v_type:
                recommendations.append(
                    {
                        "priority": "CRITICAL",
                        "action": "Investigate chain of custody and evidence integrity. This could be grounds for severe sanctions or case dismissal.",
                        "reasoning": f"Potential {v_type}.",
                    }
                )

        if (
            not recommendations and violations
        ):  # Generic if no specific high-priority ones triggered
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "action": "Review all detected violations in context of the full case file to determine appropriate legal strategy.",
                    "reasoning": f"{len(violations)} potential violation(s) of varying severity detected.",
                }
            )
        return recommendations
