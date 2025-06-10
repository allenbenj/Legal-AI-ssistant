"""Placeholder node classes for real-time workflow tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class DocumentProcessingNode:
    processor: Callable[[str], Any] | None = None

    async def __call__(self, document_path: str) -> Any:  # pragma: no cover - stub
        if self.processor:
            return await self.processor(document_path)
        return {}


@dataclass
class DocumentRewritingNode:
    rewriter: Callable[[str], Any] | None = None

    async def __call__(self, text: str) -> Any:  # pragma: no cover - stub
        if self.rewriter:
            return await self.rewriter(text)
        return {}


@dataclass
class HybridExtractionNode:
    extractor: Callable[[str], Any] | None = None

    async def __call__(self, document_path: str) -> Any:  # pragma: no cover - stub
        if self.extractor:
            return await self.extractor(document_path)
        return {}


@dataclass
class OntologyExtractionNode:
    extractor: Callable[[str], Any] | None = None

    async def __call__(self, document_path: str) -> Any:  # pragma: no cover - stub
        if self.extractor:
            return await self.extractor(document_path)
        return {}


@dataclass
class GraphBuildingNode:
    builder: Callable[..., Any] | None = None

    async def __call__(self, data: Any) -> Any:  # pragma: no cover - stub
        if self.builder:
            return await self.builder(data)
        return {}


@dataclass
class VectorStoreUpdateNode:
    updater: Callable[..., Any] | None = None

    async def __call__(self, data: Any) -> Any:  # pragma: no cover - stub
        if self.updater:
            return await self.updater(data)
        return {}


@dataclass
class MemoryIntegrationNode:
    integrator: Callable[..., Any] | None = None

    async def __call__(self, data: Any) -> Any:  # pragma: no cover - stub
        if self.integrator:
            return await self.integrator(data)
        return {}


@dataclass
class ValidationNode:
    validator: Callable[..., Any] | None = None

    async def __call__(self, data: Any) -> Any:  # pragma: no cover - stub
        if self.validator:
            return await self.validator(data)
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
