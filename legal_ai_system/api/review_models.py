from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..utils.reviewable_memory import ReviewStatus, ReviewPriority


class ReviewItem(BaseModel):
    """Representation of an item awaiting review."""

    item_id: str
    item_type: str
    content: Dict[str, Any]
    confidence: float
    source_document_id: str
    extraction_context: Optional[Dict[str, Any]] = None
    review_status: ReviewStatus
    review_priority: ReviewPriority
    created_at: datetime
    reviewed_at: Optional[datetime] = None
    reviewer_id: Optional[str] = None
    reviewer_notes: Optional[str] = None


class ReviewDecisionRequest(BaseModel):
    """Payload for submitting a review decision."""

    item_id: str
    decision: ReviewStatus
    modified_content: Optional[Dict[str, Any]] = None
    reviewer_notes: Optional[str] = None
    confidence_override: Optional[float] = Field(None, ge=0.0, le=1.0)
    reviewer_id: Optional[str] = None


class ReviewDecisionResponse(BaseModel):
    status: str
    item_id: str


class ReviewStatsResponse(BaseModel):
    status_counts: Dict[str, int]
    priority_counts_pending: Dict[str, int]
    pending_reviews_total: int
    new_items_last_24h: int
    auto_approve_thresh: float
    review_thresh: float
    reject_thresh: float
    db_path: str
