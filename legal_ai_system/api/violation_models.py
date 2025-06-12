from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ViolationEntry(BaseModel):
    """Representation of a violation record."""

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
    recommended_motion: Optional[str] = None


class ViolationStatusUpdate(BaseModel):
    """Payload for updating violation status."""

    status: str
    reviewed_by: Optional[str] = None
