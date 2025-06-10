from __future__ import annotations

from typing import Any, Dict, List

try:  # pragma: no cover - optional dependency
    from langgraph.graph import BaseNode
except Exception:  # pragma: no cover - fallback stub
    class BaseNode:
        pass

from ...utils.reviewable_memory import ReviewableMemory, ReviewPriority


class HumanReviewNode(BaseNode):
    """Node that sends extraction results to ``ReviewableMemory`` and returns high-risk items."""

    def __init__(self, review_memory: ReviewableMemory) -> None:
        self.review_memory = review_memory

    async def __call__(self, extraction: Any) -> Dict[str, Any]:
        document_id = getattr(extraction, "document_id", "")
        stats = await self.review_memory.process_extraction_result(extraction, document_id)
        critical = await self.review_memory.get_pending_reviews_async(priority=ReviewPriority.CRITICAL)
        high = await self.review_memory.get_pending_reviews_async(priority=ReviewPriority.HIGH)
        items: List[Any] = [*critical, *high]
        return {
            "stats": stats,
            "high_risk_findings": [i.to_dict() for i in items],
        }


__all__ = ["HumanReviewNode"]
