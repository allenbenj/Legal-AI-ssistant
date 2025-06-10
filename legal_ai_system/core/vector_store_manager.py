"""Simple Vector Store Manager interface."""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional

try:
    from .vector_store import VectorStore, create_vector_store
except Exception:  # pragma: no cover - optional dependency may be missing
    VectorStore = Any  # type: ignore[assignment]
    create_vector_store = None


class VectorStoreManager:
    """Lightweight manager around :class:`VectorStore`."""

    def __init__(self, store: Optional[Any] = None) -> None:
        self.store = store
        self._initialized = False

    async def initialize(self, service_container: Optional[Any] = None) -> None:
        """Initialize the underlying vector store if needed."""
        if self.store is None and create_vector_store is not None:
            self.store = create_vector_store(service_container or {})
        if self.store and hasattr(self.store, "initialize") and not self._initialized:
            await self.store.initialize()
            self._initialized = True

    async def add_texts(self, texts: List[str]) -> None:
        if not self.store:
            raise RuntimeError("Vector store not initialized")
        if hasattr(self.store, "add_texts"):
            await self.store.add_texts(texts)

    async def search(self, query: str, top_k: int = 5) -> List[Any]:
        if not self.store:
            return []
        if hasattr(self.store, "search"):
            return await self.store.search(query, top_k=top_k)
        return []


def create_vector_store_manager(service_container: Optional[Any] = None) -> VectorStoreManager:
    """Factory for :class:`VectorStoreManager`."""
    manager = VectorStoreManager()
    if service_container is not None and create_vector_store is not None:
        asyncio.run(manager.initialize(service_container))
    return manager
