from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class CaseWorkflowState:
    """Simple container tracking documents and context for a case."""

    case_id: str
    documents: Dict[str, str] = field(default_factory=dict)
    document_order: List[str] = field(default_factory=list)
    state_data: Dict[str, Any] = field(default_factory=dict)

    def process_new_document(self, document_id: str, document_text: str) -> None:
        """Store the text for a processed document in submission order."""
        self.documents[document_id] = document_text
        self.document_order.append(document_id)

    def get_case_context(self) -> str:
        """Aggregate text from all processed documents in order."""
        return "\n".join(self.documents[doc_id] for doc_id in self.document_order)

    def update_case_state(self, new_data: Dict[str, Any]) -> None:
        """Merge additional metadata into the case state."""
        self.state_data.update(new_data)


__all__ = ["CaseWorkflowState"]
