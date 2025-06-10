from __future__ import annotations

"""Compatibility wrappers around :class:`EnhancedVectorStore`."""

from typing import Any, Dict, List, Optional
import hashlib

from ..core.enhanced_vector_store import EnhancedVectorStore, IndexType


class VectorStoreEnhanced(EnhancedVectorStore):
    """Backward-compatible alias delegating to :class:`EnhancedVectorStore`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


def create_vector_store_enhanced(
    service_container: Any,
    service_config: Optional[Dict[str, Any]] | None = None,
) -> VectorStoreEnhanced:
    """Factory matching :func:`create_enhanced_vector_store`."""

    cfg = service_config or {}
    return VectorStoreEnhanced(
        storage_path=cfg.get("STORAGE_PATH", "./storage/vectors"),
        embedding_model=cfg.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        index_type=IndexType(cfg.get("DEFAULT_INDEX_TYPE", "HNSW")),
        enable_gpu=cfg.get("ENABLE_GPU_FAISS", False),
    )


class EmbeddingClient:
    """Minimal embedding client using a stable hash as pseudo-embedding."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).hexdigest()
            vector = [int(digest[i:i+8], 16) / 1e8 for i in range(0, 32, 8)]
            embeddings.append(vector)
        return embeddings


class MemoryStore:
    """Placeholder memory store for backward compatibility."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
