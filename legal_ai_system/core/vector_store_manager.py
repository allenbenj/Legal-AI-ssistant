"""Simple VectorStoreManager abstraction."""

from typing import Any, Dict, List, Optional


class VectorStoreManager:
    """Minimal manager class wrapping a vector store implementation."""

    def __init__(self, vector_store: Optional[Any] = None) -> None:
        self.vector_store = vector_store

    def set_vector_store(self, vector_store: Any) -> None:
        """Attach a vector store instance."""
        self.vector_store = vector_store

    def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Proxy to add vectors to the underlying store."""
        if self.vector_store and hasattr(self.vector_store, "add_vectors"):
            return self.vector_store.add_vectors(vectors, metadata)
        return None

    def search(self, vector: List[float], k: int = 5):
        """Proxy search to the underlying store."""
        if self.vector_store and hasattr(self.vector_store, "search"):
            return self.vector_store.search(vector, k)
        return []


