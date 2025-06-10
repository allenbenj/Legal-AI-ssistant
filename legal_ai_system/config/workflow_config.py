from __future__ import annotations

"""Workflow configuration specifying agent components used in workflows."""

from dataclasses import dataclass, field
from typing import Dict, Type

from .agent_unified_config import AGENT_CLASS_REGISTRY




@dataclass
class WorkflowConfig:
    """Configuration for selecting agent implementations for workflow steps."""

    document_processor: str = "DocumentProcessorAgent"
    document_rewriter: str = "DocumentRewriterAgent"
    ontology_extractor: str = "OntologyExtractionAgent"
    hybrid_extractor: str = "HybridLegalExtractor"
    graph_manager: str = "RealTimeGraphManager"
    vector_store: str = "OptimizedVectorStore"
    reviewable_memory: str = "ReviewableMemory"

    extra_options: Dict[str, object] = field(default_factory=dict)

    def resolve_class(self, component_name: str) -> Type:
        """Return the class configured for the given component."""
        identifier = getattr(self, component_name)
        if identifier not in AGENT_CLASS_REGISTRY:
            raise ValueError(f"Unknown component identifier: {identifier}")
        return AGENT_CLASS_REGISTRY[identifier]
