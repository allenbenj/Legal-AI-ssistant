from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from ..utils.reviewable_memory import ReviewStatus


class ViolationRecord(BaseModel):
    """Violation record returned by the API."""

    id: str
    document_id: str
    violation_type: str
    severity: str
    status: str
    description: str
    confidence: float
    detected_time: datetime
    reviewed_by: Optional[str] = None
    review_time: Optional[datetime] = None
    comments: Optional[str] = None


class ViolationDecisionRequest(BaseModel):
    """Request payload for submitting a violation decision."""

    decision: ReviewStatus
    reviewer_id: Optional[str] = None


class ViolationDecisionResponse(BaseModel):
    status: str
    violation_id: str
