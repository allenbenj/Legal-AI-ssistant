from __future__ import annotations

from typing import Any, Dict, List

try:  # pragma: no cover - optional dependency during lightweight tests
    from ...utils.reviewable_memory import ReviewPriority, ReviewableMemory
except Exception:  # pragma: no cover - define minimal fallbacks
    from enum import Enum
    from typing import Protocol

    class ReviewPriority(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

    class ReviewableMemory(Protocol):  # type: ignore[misc]
        async def process_extraction_result(self, extraction: Any, doc_id: str) -> None:
            ...

        async def get_pending_reviews_async(self, *, priority: ReviewPriority) -> List[Any]:
            ...


class HumanReviewNode:
    """Persist extraction results and surface high‑risk items for review."""

    def __init__(self, review_memory: ReviewableMemory) -> None:
        self.review_memory = review_memory

    async def __call__(self, data: Any) -> Dict[str, Any]:
        """Store extraction output and return high risk findings.

        The node accepts either a dictionary containing an ``ontology_result``
        key and ``document_id`` or the raw extraction object itself.  After
        persisting the extraction to :class:`ReviewableMemory`, any pending
        critical/high priority items are fetched and returned so the caller can
        notify a human reviewer.
        """

        # Determine extraction object and document identifier
        extraction = None
        doc_id = None
        if isinstance(data, dict):
            extraction = data.get("ontology_result") or data.get("extraction_result")
            doc_id = data.get("document_id")
            if extraction is None:
                extraction = data
        else:
            extraction = data
            doc_id = getattr(data, "document_id", None)

        if extraction is not None and doc_id is not None:
            await self.review_memory.process_extraction_result(extraction, doc_id)

        high_risk: List[Dict[str, Any]] = []
        for prio in (ReviewPriority.CRITICAL, ReviewPriority.HIGH):
            pending = await self.review_memory.get_pending_reviews_async(priority=prio)
            try:
                iterator = list(pending)  # type: ignore[arg-type]
            except Exception:
                iterator = []
            for item in iterator:
                to_add = item.to_dict() if hasattr(item, "to_dict") else item
                high_risk.append(to_add)

        # Return list for object input and count for dict input to match tests
        if isinstance(data, dict):
            result = dict(data)
            result["high_risk_findings"] = len(high_risk)
            return result
        return {"high_risk_findings": high_risk}

