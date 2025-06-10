from dataclasses import dataclass, field
from typing import Dict

from .agent_unified_config import AGENT_CLASS_MAP


@dataclass
class WorkflowConfig:
    """Configuration describing which component class to use for each workflow step."""

    document_processor_agent: str = "DocumentProcessorAgent"
    document_rewriter_agent: str = "DocumentRewriterAgent"
    ontology_extraction_agent: str = "OntologyExtractionAgent"
    hybrid_extractor: str = "HybridLegalExtractor"
    graph_manager: str = "RealTimeGraphManager"
    vector_store: str = "OptimizedVectorStore"
    reviewable_memory: str = "ReviewableMemory"

    def as_dict(self) -> Dict[str, str]:
        return {
            "document_processor_agent": self.document_processor_agent,
            "document_rewriter_agent": self.document_rewriter_agent,
            "ontology_extraction_agent": self.ontology_extraction_agent,
            "hybrid_extractor": self.hybrid_extractor,
            "graph_manager": self.graph_manager,
            "vector_store": self.vector_store,
            "reviewable_memory": self.reviewable_memory,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "WorkflowConfig":
        return cls(**data)

    def resolve_class(self, component_key: str):
        """Return the class for the given component based on current config."""
        class_name = getattr(self, component_key)
        return AGENT_CLASS_MAP.get(class_name)


__all__ = ["WorkflowConfig"]
