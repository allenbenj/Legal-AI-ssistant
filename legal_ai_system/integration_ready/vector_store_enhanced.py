"""Simplified stubs used during testing."""

from typing import Any, List


class EmbeddingClient:
    """Minimal embedding client placeholder."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or "dummy"

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * 3 for _ in texts]


class MemoryStore:
    """Minimal memory store placeholder."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.storage: List[Any] = []

    async def add(self, item: Any) -> None:
        self.storage.append(item)
