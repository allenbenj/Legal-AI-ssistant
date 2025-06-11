"""Workflow utilities and pipeline helpers."""

from .agent_workflow import AgentWorkflow
from .legal_workflow_builder import LegalWorkflowBuilder

from ..workflow_engine.merge import (
    MergeStrategy,
    ConcatMerge,
    ListMerge,
    DictUpdateMerge,
)

from .workflow_policy import WorkflowPolicy

__all__ = [
    "AgentWorkflow",
    "LegalWorkflowBuilder",
    "build_advanced_legal_workflow",
    "MergeStrategy",
    "ConcatMerge",
    "ListMerge",
    "WorkflowPolicy",
]
