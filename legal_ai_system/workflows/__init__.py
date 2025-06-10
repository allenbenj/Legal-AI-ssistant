"""Workflow utilities and pipeline helpers."""

from .agent_workflow import AgentWorkflow
from .legal_workflow_builder import LegalWorkflowBuilder
from .routing.advanced_workflow import build_advanced_legal_workflow
from ..workflow_engine.merge import (
    MergeStrategy,
    ConcatMerge,
    ListMerge,
    DictUpdateMerge,
)


__all__ = [
    "AgentWorkflow",
    "LegalWorkflowBuilder",
    "build_advanced_legal_workflow",
    "MergeStrategy",
    "ConcatMerge",
    "ListMerge",

]
