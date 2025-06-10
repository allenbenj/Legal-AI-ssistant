from __future__ import annotations

"""Compatibility wrapper around :class:`EnhancedVectorStore`."""

from typing import Any, Dict, Optional

from .enhanced_vector_store import EnhancedVectorStore, IndexType


class OptimizedVectorStore(EnhancedVectorStore):
    """Thin wrapper that delegates to :class:`EnhancedVectorStore`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


def create_optimized_vector_store(
    service_container: Any, service_config: Optional[Dict[str, Any]] | None = None
) -> OptimizedVectorStore:
    """Factory returning an :class:`OptimizedVectorStore` instance."""

    cfg = service_config or {}
    return OptimizedVectorStore(
        storage_path=cfg.get("STORAGE_PATH", "./storage/vectors"),
        embedding_model=cfg.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        index_type=IndexType(cfg.get("DEFAULT_INDEX_TYPE", "HNSW")),
        enable_gpu=cfg.get("ENABLE_GPU_FAISS", False),
        index_params=cfg.get("INDEX_PARAMS"),
    )

