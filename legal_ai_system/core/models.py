# legal_ai_system/core/models.py
"""
Core Data Models for Legal AI System
====================================
Defines fundamental data structures used throughout the Legal AI System.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


class DocumentStatus(Enum):
    """Document processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class EntityType(Enum):
    """Legal entity types."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    MONEY = "money"
    CASE = "case"
    STATUTE = "statute"
    REGULATION = "regulation"
    CONTRACT = "contract"
    CLAUSE = "clause"
    OBLIGATION = "obligation"
    VIOLATION = "violation"
    JURISDICTION = "jurisdiction"
    OTHER = "other"


class ConfidenceLevel(Enum):
    """Confidence levels for AI extractions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class LegalDocument:
    """Represents a legal document in the system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: Optional[Path] = None
    filename: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: int = 0
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Update derived fields after initialization."""
        if self.file_path and not self.filename:
            self.filename = self.file_path.name
        if self.file_path and self.file_path.exists():
            self.size_bytes = self.file_path.stat().st_size


@dataclass
class ExtractedEntity:
    """Represents an extracted legal entity."""
    text: str
    entity_type: EntityType
    confidence: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_pos: int = 0
    end_pos: int = 0
    context: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Convert numeric confidence to level."""
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


@dataclass
class ExtractedRelationship:
    """Represents a relationship between entities."""
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ProcessingResult:
    """Generic result container for processing operations."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LegalCitation:
    """Represents a legal citation."""
    text: str
    citation_type: str  # case, statute, regulation, etc.
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    jurisdiction: Optional[str] = None
    year: Optional[int] = None
    court: Optional[str] = None
    confidence: float = 1.0
    source_document_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LegalViolation:
    """Represents a detected legal violation."""
    violation_type: str
    description: str
    severity: str  # low, medium, high, critical
    confidence: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_document_id: Optional[str] = None
    related_entities: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentMemoryEntry:
    """Memory entry for agent state persistence."""
    agent_name: str
    session_id: str
    key: str
    value: Any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    importance: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class UserFeedback:
    """User feedback for model improvement."""
    item_id: str  # ID of the item being reviewed
    item_type: str  # entity, relationship, violation, etc.
    feedback_type: str  # approve, reject, modify
    original_data: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    corrected_data: Optional[Dict[str, Any]] = None
    user_notes: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Type aliases for convenience
DocumentID = str
EntityID = str
RelationshipID = str
SessionID = str
