# legal_ai_system/agents/legal_reasoning_engine.py
"""LegalReasoningEngine - specialized reasoning utilities for legal analysis.

This module defines dataclasses for different types of legal reasoning
results and a :class:`LegalReasoningEngine` that orchestrates LLM-based
reasoning operations. The design mirrors existing agents and uses
:class:`LLMManager` for language model interaction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..core.base_agent import BaseAgent
from ..core.llm_providers import LLMManager, LLMProviderEnum, LLMProviderError
from ..utils.json_utils import extract_json_from_llm_response


@dataclass
class AnalogicalAnalysis:
    """Results from analogical reasoning comparing case facts to precedent."""

    analogies: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_summary: str = ""
    confidence_score: float = 0.0
    processing_time_sec: float = 0.0
    model_used: str = ""
    errors: List[str] = field(default_factory=list)
    analyzed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StatutoryInterpretation:
    """Results from statutory interpretation analysis."""

    statute_summary: str = ""
    interpretation: str = ""
    supporting_precedent: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time_sec: float = 0.0
    model_used: str = ""
    errors: List[str] = field(default_factory=list)
    interpreted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConstitutionalAnalysis:
    """Results from constitutional law reasoning."""

    issue_summary: str = ""
    relevant_cases: List[str] = field(default_factory=list)
    reasoning_summary: str = ""
    confidence_score: float = 0.0
    processing_time_sec: float = 0.0
    model_used: str = ""
    errors: List[str] = field(default_factory=list)
    analyzed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CaseOutcomePrediction:
    """Prediction of case outcome based on provided facts."""

    predicted_outcome: str = ""
    probability: float = 0.0
    key_factors: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time_sec: float = 0.0
    model_used: str = ""
    errors: List[str] = field(default_factory=list)
    predicted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LegalReasoningEngine(BaseAgent):
    """Collection of advanced legal reasoning helpers."""

    def __init__(self, service_container: Any, **config: Any) -> None:
        super().__init__(
            service_container, name="LegalReasoningEngine", agent_type="reasoning"
        )
        self.llm_manager: Optional[LLMManager] = self.get_llm_manager()
        self.llm_config = self.get_optimized_llm_config()
        self.config.update(config)
        if not self.llm_manager:
            self.logger.error(
                "LLMManager service not available. Reasoning features disabled."
            )
        self.logger.info(
            "LegalReasoningEngine initialized.",
            parameters={"model": self.llm_config.get("llm_model", "default")},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON payload from LLM response."""
        try:
            parsed = extract_json_from_llm_response(content)
            return parsed or {}
        except Exception as e:  # pragma: no cover - safety net
            self.logger.warning("Failed to parse LLM response.", exception=e)
            return {}

    async def _complete(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Helper to send prompt to LLM and parse JSON response."""
        if not self.llm_manager:
            return None
        model_to_use = self.llm_manager.primary_config.model
        provider_to_use = self.llm_manager.primary_config.provider.value
        try:
            llm_resp = await self.llm_manager.complete(
                prompt=prompt,
                model=model_to_use,
                provider=LLMProviderEnum(provider_to_use),
                temperature=0.2,
                max_tokens=1500,
            )
            return self._parse_response(llm_resp.content)
        except LLMProviderError as e:
            self.logger.error("LLMProviderError during reasoning.", exception=e)
            return {"errors": [f"LLM error: {str(e)}"]}
        except Exception as e:  # pragma: no cover - catch-all
            self.logger.error("Unexpected error during reasoning.", exception=e)
            return {"errors": [f"Unexpected error: {str(e)}"]}

    # ------------------------------------------------------------------
    # Public reasoning methods
    # ------------------------------------------------------------------
    async def analogical_reasoning(
        self,
        case_facts: str,
        precedent_summary: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AnalogicalAnalysis:
        """Perform analogical reasoning against precedent cases."""
        start = datetime.now(timezone.utc)
        if not case_facts:
            return AnalogicalAnalysis(errors=["No case facts provided."])
        prompt = (
            "Perform analogical legal reasoning comparing the following case facts to precedent cases.\n"
            "CASE FACTS:\n{facts}\nPRECEDENT SUMMARY:\n{precedent}\n"
            "Return JSON with keys: analogies (list), reasoning_summary, confidence."
        ).format(facts=case_facts, precedent=precedent_summary)
        data = await self._complete(prompt) or {}
        result = AnalogicalAnalysis(
            analogies=data.get("analogies", []),
            reasoning_summary=data.get("reasoning_summary", ""),
            confidence_score=float(data.get("confidence", 0.0)),
            processing_time_sec=(datetime.now(timezone.utc) - start).total_seconds(),
            model_used=(
                self.llm_manager.primary_config.model if self.llm_manager else ""
            ),
            errors=data.get("errors", []),
        )
        return result

    async def statutory_interpretation(
        self,
        statute_text: str,
        question: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StatutoryInterpretation:
        """Interpret a statute in relation to a question or scenario."""
        start = datetime.now(timezone.utc)
        if not statute_text:
            return StatutoryInterpretation(errors=["No statute text provided."])
        prompt = (
            "Interpret the following statute and answer any specific question.\n"
            "STATUTE:\n{statute}\nQUESTION:\n{question}\n"
            "Return JSON with keys: statute_summary, interpretation, supporting_precedent, confidence."
        ).format(statute=statute_text, question=question)
        data = await self._complete(prompt) or {}
        result = StatutoryInterpretation(
            statute_summary=data.get("statute_summary", ""),
            interpretation=data.get("interpretation", ""),
            supporting_precedent=data.get("supporting_precedent", []),
            confidence_score=float(data.get("confidence", 0.0)),
            processing_time_sec=(datetime.now(timezone.utc) - start).total_seconds(),
            model_used=(
                self.llm_manager.primary_config.model if self.llm_manager else ""
            ),
            errors=data.get("errors", []),
        )
        return result

    async def constitutional_analysis(
        self, issue_text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ConstitutionalAnalysis:
        """Analyze constitutional issues in the given text."""
        start = datetime.now(timezone.utc)
        if not issue_text:
            return ConstitutionalAnalysis(errors=["No issue text provided."])
        prompt = (
            "Provide constitutional analysis of the following issue.\n"
            "ISSUE:\n{issue}\n"
            "Return JSON with keys: relevant_cases, reasoning_summary, confidence."
        ).format(issue=issue_text)
        data = await self._complete(prompt) or {}
        result = ConstitutionalAnalysis(
            issue_summary=issue_text[:200],
            relevant_cases=data.get("relevant_cases", []),
            reasoning_summary=data.get("reasoning_summary", ""),
            confidence_score=float(data.get("confidence", 0.0)),
            processing_time_sec=(datetime.now(timezone.utc) - start).total_seconds(),
            model_used=(
                self.llm_manager.primary_config.model if self.llm_manager else ""
            ),
            errors=data.get("errors", []),
        )
        return result

    async def predict_case_outcome(
        self, case_facts: str, metadata: Optional[Dict[str, Any]] = None
    ) -> CaseOutcomePrediction:
        """Predict the likely outcome of a case based on its facts."""
        start = datetime.now(timezone.utc)
        if not case_facts:
            return CaseOutcomePrediction(errors=["No case facts provided."])
        prompt = (
            "Predict the outcome of the case described below.\n"
            "CASE FACTS:\n{facts}\n"
            "Return JSON with keys: predicted_outcome, probability, key_factors, confidence."
        ).format(facts=case_facts)
        data = await self._complete(prompt) or {}
        result = CaseOutcomePrediction(
            predicted_outcome=data.get("predicted_outcome", ""),
            probability=float(data.get("probability", 0.0)),
            key_factors=data.get("key_factors", []),
            confidence_score=float(data.get("confidence", 0.0)),
            processing_time_sec=(datetime.now(timezone.utc) - start).total_seconds(),
            model_used=(
                self.llm_manager.primary_config.model if self.llm_manager else ""
            ),
            errors=data.get("errors", []),
        )
        return result
