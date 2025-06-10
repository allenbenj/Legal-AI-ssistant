"""Minimal async node stubs used in real-time workflows."""
from __future__ import annotations

from typing import Any

from ..workflow_engine.types import LegalWorkflowNode


class DocumentProcessingNode:
    """Placeholder for document parsing logic."""

    async def __call__(self, path: str) -> Any:  # pragma: no cover - simple stub
        """Process a document located at ``path``."""
        return {"processed_path": path}


class DocumentRewritingNode:
    """Placeholder for document rewriting logic."""

    async def __call__(self, text: str) -> Any:  # pragma: no cover - simple stub
        """Rewrite the supplied ``text``."""
        return text


class HybridExtractionNode:
    """Placeholder for hybrid entity extraction logic."""

    async def __call__(self, text: str) -> Any:  # pragma: no cover - simple stub
        """Extract entities from ``text``."""
        return {"hybrid_extraction": text}


class OntologyExtractionNode:
    """Placeholder for ontology extraction logic."""

    async def __call__(self, text: str) -> Any:  # pragma: no cover - simple stub
        """Extract ontology items from ``text``."""
        return {"ontology_extraction": text}


class GraphBuildingNode:
    """Placeholder for graph building logic."""

    async def __call__(self, data: Any) -> Any:  # pragma: no cover - simple stub
        """Build a knowledge graph from ``data``."""
        return {"graph": data}


class VectorStoreUpdateNode:
    """Placeholder for vector store update logic."""

    async def __call__(self, data: Any) -> Any:  # pragma: no cover - simple stub
        """Update vector store entries using ``data``."""
        return {"vector_update": data}


class MemoryIntegrationNode:
    """Placeholder for memory integration logic."""

    async def __call__(self, data: Any) -> Any:  # pragma: no cover - simple stub
        """Integrate ``data`` into long-term memory."""
        return {"memory_update": data}


class ValidationNode:
    """Placeholder for validation logic."""

    async def __call__(self, data: Any) -> Any:  # pragma: no cover - simple stub
        """Validate the provided ``data``."""
        return {"validation": data}


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
