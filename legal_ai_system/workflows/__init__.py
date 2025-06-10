"""Workflow utilities and pipeline helpers."""

from .agent_workflow import AgentWorkflow
from .legal_workflow_builder import LegalWorkflowBuilder
from .nodes.legal_error_handling_node import LegalErrorHandlingNode
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
    "LegalErrorHandlingNode",
]
