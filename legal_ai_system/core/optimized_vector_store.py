from __future__ import annotations

"""Compatibility wrapper around :class:`EnhancedVectorStore`."""

from typing import Any, Optional

from .config_models import OptimizedVectorStoreConfig

from .enhanced_vector_store import EnhancedVectorStore, IndexType


class OptimizedVectorStore(EnhancedVectorStore):
    """Thin wrapper that delegates to :class:`EnhancedVectorStore`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


def create_optimized_vector_store(
    service_container: Any,
    service_config: Optional[OptimizedVectorStoreConfig] | None = None,
) -> OptimizedVectorStore:
    """Factory returning an :class:`OptimizedVectorStore` instance."""

    cfg = service_config or OptimizedVectorStoreConfig()
    return OptimizedVectorStore(
        storage_path=str(cfg.storage_path),
        embedding_model=cfg.embedding_model_name,
        index_type=IndexType(cfg.default_index_type),
        enable_gpu=cfg.enable_gpu_faiss,
        index_params=cfg.index_params,
    )
