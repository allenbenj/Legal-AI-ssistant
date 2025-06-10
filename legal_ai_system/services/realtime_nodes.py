from __future__ import annotations

"""Lightweight node stubs used by :mod:`realtime_analysis_workflow` tests."""

from typing import Any


class DocumentProcessingNode:
    async def run(self, path: str, **kwargs: Any) -> Any:  # pragma: no cover
        return path


class DocumentRewritingNode:
    async def run(self, text: str, **kwargs: Any) -> Any:  # pragma: no cover
        return text


class HybridExtractionNode:
    async def run(self, document: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return {}


class OntologyExtractionNode:
    async def run(self, document: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return {}


class GraphBuildingNode:
    async def run(self, data: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return {}


class VectorStoreUpdateNode:
    async def run(self, data: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return {}


class MemoryIntegrationNode:
    async def run(self, data: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return {}


class ValidationNode:
    async def run(self, data: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return {}


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
