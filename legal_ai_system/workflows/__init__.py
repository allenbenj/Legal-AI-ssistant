"""Workflow utilities and pipeline helpers."""

from .agent_workflow import AgentWorkflow
from .legal_workflow_builder import LegalWorkflowBuilder


__all__ = [
    "AgentWorkflow",
    "LegalWorkflowBuilder",
    "MergeStrategy",
    "FirstResultMerge",
    "ListMerge",
    "DictMerge",

]
