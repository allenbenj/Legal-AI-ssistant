from dataclasses import dataclass, asdict
from typing import Any, Dict

from config.workflow_config import USE_WORKFLOW_BUILDER

@dataclass
class WorkflowConfig:
    """Configuration options for :class:`RealTimeAnalysisWorkflow`."""

    enable_real_time_sync: bool = True
    confidence_threshold: float = 0.75
    enable_user_feedback: bool = True
    parallel_processing: bool = True
    max_concurrent_documents: int = 3
    performance_monitoring: bool = True
    auto_optimization_threshold: int = 100
    # Toggle between legacy workflow and new builder implementation
    use_builder: bool = USE_WORKFLOW_BUILDER

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = ["WorkflowConfig", "USE_WORKFLOW_BUILDER"]
