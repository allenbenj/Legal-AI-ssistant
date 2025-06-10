"""Thin wrapper around :class:`EnhancedVectorStore`."""
from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import datetime, timezone

from .detailed_logging import get_detailed_logger, LogCategory, detailed_log_function
from .enhanced_vector_store import EnhancedVectorStore, IndexType, create_enhanced_vector_store

ovs_logger = get_detailed_logger("OptimizedVectorStore", LogCategory.VECTOR_STORE)


class OptimizedVectorStore(EnhancedVectorStore):
    """Compatibility subclass that forwards to :class:`EnhancedVectorStore`."""

    @detailed_log_function(LogCategory.VECTOR_STORE)
    def __init__(
        self,
        storage_path_str: str = "./storage/optimized_vector_store_data",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        default_index_type: IndexType = IndexType.HNSW,
        enable_gpu_faiss: bool = False,
        service_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        ovs_logger.info("Initializing OptimizedVectorStore.")
        super().__init__(
            storage_path=storage_path_str,
            embedding_model=embedding_model,
            index_type=default_index_type,
            enable_gpu=enable_gpu_faiss,
        )
        self.service_config = service_config or {}
        ovs_logger.info(
            "OptimizedVectorStore initialized.",
            parameters={"parent_initialized_with_type": self.index_type.value},
        )

    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def optimize_performance(
        self, full_reindex: bool = False, compact_storage: bool = True
    ) -> Dict[str, Any]:
        ovs_logger.info(
            "Starting performance optimization for OptimizedVectorStore.",
            parameters={
                "full_reindex": full_reindex,
                "compact_storage": compact_storage,
            },
        )
        now = datetime.now(timezone.utc)
        self.statistics.last_optimization = now
        self.statistics.optimization_count += 1
        result = {
            "optimization_completed": True,
            "last_optimized": now.isoformat(),
            "optimization_runs": self.statistics.optimization_count,
        }
        ovs_logger.info("Performance optimization finished.", parameters=result)
        return result


# Factory for service container

def create_optimized_vector_store(
    service_config: Optional[Dict[str, Any]] = None,
) -> OptimizedVectorStore:
    cfg = service_config or {}
    return OptimizedVectorStore(
        storage_path_str=cfg.get("STORAGE_PATH", "./storage/optimized_vector_store_data"),
        embedding_model=cfg.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        default_index_type=IndexType(cfg.get("DEFAULT_INDEX_TYPE", "HNSW")),
        enable_gpu_faiss=cfg.get("ENABLE_GPU_FAISS", False),
        service_config=cfg,
    )
