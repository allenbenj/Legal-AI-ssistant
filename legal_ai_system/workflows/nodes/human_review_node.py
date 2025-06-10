from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

try:
    from ...utils.reviewable_memory import ReviewableMemory
except Exception:  # pragma: no cover - during tests utils may be missing
    ReviewableMemory = Any  # type: ignore


@dataclass
class HumanReviewNode:
    """Node that queues high-risk findings for human review."""

    review_memory: ReviewableMemory

    async def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        extraction = data.get("ontology_result")
        document_id = data.get("document_id", "unknown")
        if not extraction:
            return data

        high_risk_count = 0
        for ent in getattr(extraction, "entities", []):
            text = getattr(ent, "source_text_snippet", "").lower()
            etype = getattr(ent, "entity_type", "").lower()
            if any(k in text for k in ["violation", "misconduct", "fraud", "brady"]) or (
                "violation" in etype or "misconduct" in etype
            ):
                high_risk_count += 1
        for rel in getattr(extraction, "relationships", []):
            text = getattr(rel, "evidence_text_snippet", "").lower()
            rtype = getattr(rel, "relationship_type", "").lower()
            if any(k in text for k in ["violation", "misconduct", "fraud", "brady"]) or (
                "violation" in rtype or "misconduct" in rtype
            ):
                high_risk_count += 1

        meta: Dict[str, Any] | None = None
        if high_risk_count:
            meta = {"high_risk_findings": high_risk_count}
            data["high_risk_findings"] = high_risk_count

        await self.review_memory.process_extraction_result(
            extraction, document_id, meta
        )
        return data
