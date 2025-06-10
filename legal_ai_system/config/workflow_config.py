
@dataclass
class WorkflowConfig:
    hybrid_extractor: str = "HybridLegalExtractor"
    graph_manager: str = "RealTimeGraphManager"
    vector_store: str = "EnhancedVectorStore"
    reviewable_memory: str = "ReviewableMemory"


