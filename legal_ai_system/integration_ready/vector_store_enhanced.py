from __future__ import annotations

"""Compatibility wrappers around :class:`EnhancedVectorStore`."""

from typing import List
import hashlib

from ..core.enhanced_vector_store import EnhancedVectorStore, IndexType


class VectorStoreEnhanced(EnhancedVectorStore):
    """Backward-compatible alias delegating to :class:`EnhancedVectorStore`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EmbeddingClient:
    """Minimal embedding client using a stable hash as pseudo-embedding."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).hexdigest()
            # Convert hex digest into a small deterministic vector
            vector = [int(digest[i:i+8], 16) / 1e8 for i in range(0, 32, 8)]
            embeddings.append(vector)
        return embeddings


class MemoryStore:
    """Placeholder memory store for backward compatibility."""

    def __init__(self, db_path: str):
        self.db_path = db_path

