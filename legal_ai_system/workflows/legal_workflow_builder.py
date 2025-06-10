from __future__ import annotations

"""Convenience wrapper around :mod:`workflow_engine` builder utilities."""

from ..workflow_engine.builder import LegalWorkflowBuilder as _BaseBuilder


class LegalWorkflowBuilder(_BaseBuilder):
    """Workflow builder for orchestrating async processing steps."""

    pass


__all__ = ["LegalWorkflowBuilder"]
