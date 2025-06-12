"""Workflow utilities and pipeline helpers."""

from .agent_workflow import AgentWorkflow
from .legal_workflow_builder import LegalWorkflowBuilder

from ..workflow_engine.merge import (
    MergeStrategy,
    ConcatMerge,
    ListMerge,
    DictUpdateMerge,
)

try:  # pragma: no cover - optional dependencies
    from .workflow_policy import WorkflowPolicy
except Exception:  # pragma: no cover - gracefully handle missing ML deps
    WorkflowPolicy = None  # type: ignore[misc]

from importlib import import_module as _import_module

# Re-export nodes subpackage for convenient access
nodes = _import_module(".nodes", __name__)

__all__ = [
    "AgentWorkflow",
    "LegalWorkflowBuilder",
    "build_advanced_legal_workflow",
    "MergeStrategy",
    "ConcatMerge",
    "ListMerge",
    "DictUpdateMerge",
    "WorkflowPolicy",
    "nodes",
]
