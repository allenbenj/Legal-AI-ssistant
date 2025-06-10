"""Minimal placeholder nodes for real-time workflow tests."""
from __future__ import annotations


class LegalWorkflowNode:
    """Base placeholder class for workflow nodes."""

    async def __call__(self, *args, **kwargs):
        return None


class DocumentProcessingNode(LegalWorkflowNode):
    pass


class DocumentRewritingNode(LegalWorkflowNode):
    pass


class HybridExtractionNode(LegalWorkflowNode):
    pass


class OntologyExtractionNode(LegalWorkflowNode):
    pass


class GraphBuildingNode(LegalWorkflowNode):
    pass


class VectorStoreUpdateNode(LegalWorkflowNode):
    pass


class MemoryIntegrationNode(LegalWorkflowNode):
    pass


class ValidationNode(LegalWorkflowNode):
    pass


__all__ = [
    "LegalWorkflowNode",
    "DocumentProcessingNode",
    "DocumentRewritingNode",
    "HybridExtractionNode",
    "OntologyExtractionNode",
    "GraphBuildingNode",
    "VectorStoreUpdateNode",
    "MemoryIntegrationNode",
    "ValidationNode",
]
