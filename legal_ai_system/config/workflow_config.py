
"""Configuration options for building analysis workflows."""

from dataclasses import dataclass

# Flag to toggle between the legacy workflow implementation and the
# experimental builder-based workflow. Default is ``False`` to maintain
# current behaviour.
USE_WORKFLOW_BUILDER: bool = False


@dataclass
class WorkflowConfig:
    """Component configuration for the real-time workflow."""

    hybrid_extractor: str = "HybridLegalExtractor"
    graph_manager: str = "RealTimeGraphManager"
    vector_store: str = "EnhancedVectorStore"
    reviewable_memory: str = "ReviewableMemory"
    #: Enable the new workflow builder implementation instead of the legacy
    #: implementation. Defaults to ``False`` to preserve current behaviour.
    use_builder: bool = USE_WORKFLOW_BUILDER


