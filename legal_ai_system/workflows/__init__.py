"""Workflow utilities and pipeline helpers."""

from .agent_workflow import AgentWorkflow
from .legal_workflow_builder import LegalWorkflowBuilder
from .merge import DictMerge, FirstResultMerge, ListMerge, MergeStrategy

__all__ = [
    "AgentWorkflow",
    "LegalWorkflowBuilder",
    "MergeStrategy",
    "FirstResultMerge",
    "ListMerge",
    "DictMerge",
]
