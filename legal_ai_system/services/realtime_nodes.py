"""Minimal node definitions for the real-time workflow."""
from __future__ import annotations

from typing import Any


class DocumentProcessingNode:
    """Extract text and metadata from a document path."""

    async def __call__(self, path: str) -> Any:
        """Process ``path`` and return any extracted data."""
        ...


class DocumentRewritingNode:
    """Rewrite or clean up raw document text."""

    async def __call__(self, text: str) -> Any:
        """Return the rewritten form of ``text``."""
        ...


class HybridExtractionNode:
    """Perform hybrid entity extraction on a document."""

    async def __call__(self, path: str) -> Any:
        """Extract entities from ``path`` using hybrid techniques."""
        ...


class OntologyExtractionNode:
    """Run ontology-based extraction on a document."""

    async def __call__(self, path: str) -> Any:
        """Extract entities from ``path`` based on ontology rules."""
        ...


class GraphBuildingNode:
    """Update the knowledge graph with extracted data."""

    async def __call__(self, data: Any) -> Any:
        """Create or update graph objects using ``data``."""
        ...


class VectorStoreUpdateNode:
    """Persist embeddings or vectors for later retrieval."""

    async def __call__(self, data: Any) -> Any:
        """Update the vector store with ``data``."""
        ...


class MemoryIntegrationNode:
    """Integrate processing results with long term memory."""

    async def __call__(self, data: Any) -> Any:
        """Store ``data`` in the memory system."""
        ...


class ValidationNode:
    """Validate extracted information and produce metrics."""

    async def __call__(self, data: Any) -> Any:
        """Return validation results for ``data``."""
        ...


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
