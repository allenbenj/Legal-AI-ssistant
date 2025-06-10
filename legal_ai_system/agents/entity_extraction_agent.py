# legal_ai_system/agents/entity_extraction_streamlined/entity_extraction_agent.py
"""
Streamlined Entity Extraction Agent
===================================
Lightweight entity extraction focused on legal documents.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..core.base_agent import BaseAgent, AgentResult
from ..core.detailed_logging import (
    get_detailed_logger,
    LogCategory,
    detailed_log_function,
)
from ..core.models import LegalDocument
from ..core.unified_exceptions import AgentError


# Initialize logger
entity_extraction_logger = get_detailed_logger(
    "StreamlinedEntityExtractionAgent", LogCategory.AGENT
)


@dataclass
class LegalEntity:
    """Lightweight legal entity representation."""

    text: str
    entity_type: str
    confidence: float
    start_pos: int = 0
    end_pos: int = 0
    context: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamlinedEntityExtractionOutput:
    """Output from streamlined entity extraction."""

    document_id: str
    entities: List[LegalEntity] = field(default_factory=list)
    processing_time: float = 0.0
    confidence_threshold: float = 0.5
    total_entities_found: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        self.total_entities_found = len(self.entities)


class StreamlinedEntityExtractionAgent(BaseAgent):
    """
    Streamlined agent for extracting legal entities from documents.
    """

    @detailed_log_function(LogCategory.AGENT)
    def __init__(self, service_container: Optional[Any] = None, **kwargs):
        super().__init__(
            service_container,
            name="StreamlinedEntityExtractionAgent",
            agent_type="extraction",
        )

        self.confidence_threshold = kwargs.get("confidence_threshold", 0.5)
        self.enable_context = kwargs.get("enable_context", True)

        entity_extraction_logger.info(
            "StreamlinedEntityExtractionAgent initialized",
            parameters={"threshold": self.confidence_threshold},
        )

    @detailed_log_function(LogCategory.AGENT)
    async def extract_entities(
        self, document: LegalDocument
    ) -> StreamlinedEntityExtractionOutput:
        """Extract entities from a legal document."""
        start_time = datetime.now(timezone.utc)

        try:
            entity_extraction_logger.info(
                "Starting entity extraction", parameters={"document_id": document.id}
            )

            entities = []

            if document.content:
                # Simple pattern-based extraction (placeholder)
                entities = self._extract_with_patterns(document.content)

            # Filter by confidence threshold
            filtered_entities = [
                e for e in entities if e.confidence >= self.confidence_threshold
            ]

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            result = StreamlinedEntityExtractionOutput(
                document_id=document.id,
                entities=filtered_entities,
                processing_time=processing_time,
                confidence_threshold=self.confidence_threshold,
            )

            entity_extraction_logger.info(
                "Entity extraction completed",
                parameters={
                    "document_id": document.id,
                    "entities_found": len(filtered_entities),
                    "processing_time": processing_time,
                },
            )

            return result

        except Exception as e:
            entity_extraction_logger.error(
                "Entity extraction failed",
                parameters={"document_id": document.id},
                exception=e,
            )
            raise AgentError(f"Entity extraction failed: {str(e)}")

    def _extract_with_patterns(self, text: str) -> List[LegalEntity]:
        """Extract entities using simple patterns."""
        entities = []
        text_lower = text.lower()

        # Simple pattern matching (placeholder implementation)
        if "contract" in text_lower:
            entities.append(
                LegalEntity(text="contract", entity_type="CONTRACT", confidence=0.8)
            )

        if "agreement" in text_lower:
            entities.append(
                LegalEntity(text="agreement", entity_type="CONTRACT", confidence=0.75)
            )

        if "party" in text_lower:
            entities.append(
                LegalEntity(text="party", entity_type="ORGANIZATION", confidence=0.7)
            )

        return entities

    async def process(
        self, data: Any
    ) -> AgentResult[StreamlinedEntityExtractionOutput]:
        """BaseAgent interface implementation."""
        try:
            if isinstance(data, LegalDocument):
                result = await self.extract_entities(data)
                return AgentResult(
                    success=True,
                    data=result,
                    agent_name=self.name,
                )
            else:
                raise AgentError(
                    "Invalid input data type for StreamlinedEntityExtractionAgent"
                )

        except Exception as e:
            return AgentResult(
                success=False,
                error=str(e),
                agent_name=self.name,
            )
