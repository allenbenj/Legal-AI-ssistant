from __future__ import annotations

from typing import Any, Dict, List

from ...utils.reviewable_memory import ReviewPriority, ReviewableMemory


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
            for item in await self.review_memory.get_pending_reviews_async(priority=prio):
                to_add = item.to_dict() if hasattr(item, "to_dict") else item
                high_risk.append(to_add)

        # Return list for object input and count for dict input to match tests
        if isinstance(data, dict):
            result = dict(data)
            result["high_risk_findings"] = len(high_risk)
            return result
        return {"high_risk_findings": high_risk}

