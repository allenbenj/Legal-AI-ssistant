from __future__ import annotations

from datetime import datetime
from typing import Optional
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
