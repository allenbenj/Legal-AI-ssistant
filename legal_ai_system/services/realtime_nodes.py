"""Minimal realtime workflow node placeholders for tests."""

from __future__ import annotations

from typing import Any


class DocumentProcessingNode:
    async def run(self, data: Any) -> Any:  # pragma: no cover - placeholder
        return data


class DocumentRewritingNode(DocumentProcessingNode):
    pass


class HybridExtractionNode(DocumentProcessingNode):
    pass


class OntologyExtractionNode(DocumentProcessingNode):
    pass


class GraphBuildingNode(DocumentProcessingNode):
    pass


class VectorStoreUpdateNode(DocumentProcessingNode):
    pass


class MemoryIntegrationNode(DocumentProcessingNode):
    pass


class ValidationNode(DocumentProcessingNode):
    pass


__all__ = [
    "DocumentProcessingNode",
    "DocumentRewritingNode",
    "HybridExtractionNode",
    "OntologyExtractionNode",
    "GraphBuildingNode",
    "VectorStoreUpdateNode",
    "MemoryIntegrationNode",
    "ValidationNode",
]
