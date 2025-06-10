"""Workflow utilities and pipeline helpers."""

from .agent_workflow import AgentWorkflow
from .legal_workflow_builder import LegalWorkflowBuilder
from ..workflow_engine.merge import (
    MergeStrategy,
    ConcatMerge,
    ListMerge,
    DictUpdateMerge,
)

__all__ = [
    "AgentWorkflow",
    "LegalWorkflowBuilder",
    "MergeStrategy",
    "ConcatMerge",
    "ListMerge",
    "DictUpdateMerge",
]
