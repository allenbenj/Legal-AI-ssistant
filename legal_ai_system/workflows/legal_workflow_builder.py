"""Simplified access to the generic :class:`LegalWorkflowBuilder`."""

from __future__ import annotations

from ..workflow_engine.builder import (  # noqa: E501
    LegalWorkflowBuilder as _BaseLegalWorkflowBuilder,
)


class LegalWorkflowBuilder(_BaseLegalWorkflowBuilder):
    """Workflow builder specialized for legal processing pipelines."""

    pass


__all__ = ["LegalWorkflowBuilder"]
