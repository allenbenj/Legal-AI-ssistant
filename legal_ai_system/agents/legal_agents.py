"""Lightweight validation agents for the Violation Review GUI."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

from ..services.violation_review import ViolationReviewEntry


@dataclass
class AgentRecommendation:
    """Structured recommendation returned by a legal agent."""

    agent_name: str
    summary: str
    recommendation: Optional[str] = None
    confidence: float = 0.0
    followup_tool: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LegalAuditAgent:
    """Validate legal issues and suggest possible motions."""

    name = "LegalAuditAgent"

    def review(self, entry: ViolationReviewEntry) -> AgentRecommendation:
        desc = entry.description.lower()
        motion: Optional[str] = None
        if "perjury" in desc:
            motion = "Motion to Dismiss"
        elif "sanction" in desc:
            motion = "Motion for Sanctions"
        elif "unlawful search" in desc or "illegal search" in desc:
            motion = "Motion to Suppress Evidence"
        elif "hearsay" in desc:
            motion = "Motion to Exclude Testimony"
        summary = (
            f"Suggested motion: {motion}" if motion else "No specific motion suggested"
        )
        return AgentRecommendation(
            agent_name=self.name,
            summary=summary,
            recommendation=motion,
            confidence=entry.confidence,
        )


class EthicsReviewAgent:
    """Flag potential bar rule violations."""

    name = "EthicsReviewAgent"

    def review(self, entry: ViolationReviewEntry) -> AgentRecommendation:
        desc = entry.description.lower()
        rule = None
        if "perjury" in desc:
            rule = "NC RPC 3.3 - Candor Toward the Tribunal"
        summary = (
            f"Possible bar violation: {rule}" if rule else "No ethical issue detected"
        )
        return AgentRecommendation(
            agent_name=self.name,
            summary=summary,
            recommendation=rule,
            confidence=entry.confidence,
        )


class LEOConductAgent:
    """Detect potential law enforcement misconduct."""

    name = "LEOConductAgent"

    def review(self, entry: ViolationReviewEntry) -> AgentRecommendation:
        desc = entry.description.lower()
        violation = None
        if "false arrest" in desc or "unlawful arrest" in desc:
            violation = "IACP Policy Violation"
        summary = (
            f"Police misconduct: {violation}" if violation else "No misconduct detected"
        )
        return AgentRecommendation(
            agent_name=self.name,
            summary=summary,
            recommendation=violation,
            confidence=entry.confidence,
        )
