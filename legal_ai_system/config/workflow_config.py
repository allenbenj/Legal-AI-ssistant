
"""Configuration options for building analysis workflows."""

from dataclasses import dataclass


@dataclass
class WorkflowConfig:
    """Component configuration for the real-time workflow."""

    hybrid_extractor: str = "HybridLegalExtractor"
    graph_manager: str = "RealTimeGraphManager"
    vector_store: str = "EnhancedVectorStore"
    reviewable_memory: str = "ReviewableMemory"
    #: Enable the new workflow builder implementation instead of the legacy
    #: implementation. Defaults to ``False`` to preserve current behaviour.
    use_builder: bool = False


