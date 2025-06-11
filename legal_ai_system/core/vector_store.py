# legal_ai_system/knowledge/vector_store/vector_store.py
"""
Vector Store - Consolidated with DETAILED Logging and Full Implementation
==========================================================================
Unified vector storage combining the best features from previous versions,
with comprehensive detailed logging of every operation.
Manages FAISS indexes, metadata, caching, and performance optimization.
Utilizes an injected, pre-initialized EmbeddingProvider.
"""

import asyncio
import hashlib
import json
import logging  # For tenacity's before_sleep_log
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from collections import deque
from cachetools import LRUCache
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .vector_metadata_repository import VectorMetadataRepository

import faiss  # type: ignore
import numpy as np  # type: ignore
import tenacity  # For retries

# Core imports
from .detailed_logging import LogCategory, detailed_log_function, get_detailed_logger
from .agent_unified_config import _get_service_sync
from .unified_exceptions import ConfigurationError, DatabaseError, VectorStoreError

# Initialize loggers for this module
vector_store_logger = get_detailed_logger("VectorStore", LogCategory.VECTOR_STORE)
vs_search_logger = get_detailed_logger("VectorStore.Search", LogCategory.VECTOR_STORE)
vs_embedding_logger = get_detailed_logger(
    "VectorStore.Embedding", LogCategory.VECTOR_STORE
)
vs_index_logger = get_detailed_logger("VectorStore.IndexMgmt", LogCategory.VECTOR_STORE)
vs_cache_logger = get_detailed_logger("VectorStore.Cache", LogCategory.VECTOR_STORE)
vs_perf_logger = get_detailed_logger("VectorStore.Performance", LogCategory.PERFORMANCE)


class VectorStoreState(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    INDEXING = "indexing"
    SEARCHING = "searching"
    OPTIMIZING = "optimizing"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


class IndexType(Enum):
    FLAT = "IndexFlatL2"
    IVF = "IndexIVFFlat"
    HNSW = "IndexHNSWFlat"
    PQ = "IndexPQ"
    IVFPQ = "IndexIVFPQ"


@dataclass
class VectorMetadata:
    faiss_id: int
    vector_id: str
    document_id: str
    content_hash: str
    content_preview: str
    vector_norm: float
    dimension: int
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row_dict(cls, row_dict: Dict[str, Any]) -> "VectorMetadata":
        return cls(
            faiss_id=int(row_dict.get("faiss_id", -1)),
            vector_id=str(row_dict["vector_id"]),
            document_id=str(row_dict["document_id"]),
            content_hash=str(row_dict["content_hash"]),
            content_preview=str(row_dict.get("content_preview", "")),
            vector_norm=float(row_dict.get("vector_norm", 0.0)),
            dimension=int(row_dict.get("dimension", 0)),
        )


@dataclass
class SearchResult:
    vector_id: str
    document_id: str
    content_preview: str
    similarity_score: float
    distance: float
    metadata: VectorMetadata
    search_time_sec: float
    index_type_used: str
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metadata"] = self.metadata.to_dict()
        return data


@dataclass
class IndexStatistics:
    total_vectors: int = 0
    total_adds_session: int = 0
    avg_add_time_sec_session: float = 0.0
    total_searches_session: int = 0
    avg_search_time_sec_session: float = 0.0
    index_disk_size_mb: float = 0.0
    metadata_db_size_mb: float = 0.0  # For on-disk sizes
    cache_hit_rate_session: float = 0.0
    cache_hits_session: int = 0
    cache_misses_session: int = 0
    last_optimization_at_iso: Optional[str] = None
    optimization_runs_count: int = 0
    faiss_index_memory_mb: float = 0.0  # Approx FAISS RAM usage


class EmbeddingProviderVS(ABC):
    def __init__(
        self, model_name: str, service_config: Optional[Dict[str, Any]] = None
    ):
        self.model_name = model_name
        self.dimension: Optional[int] = None
        self.config = service_config or {}
        self.logger = vs_embedding_logger.getChild(self.__class__.__name__)

    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def embed_texts(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        pass


class SentenceTransformerEmbeddingProvider(EmbeddingProviderVS):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        service_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_name, service_config)
        self.model: Optional[Any] = None
        self._initialized_flag = False

    @detailed_log_function(LogCategory.VECTOR_STORE_EMBEDDING)
    async def initialize(self):
        if self._initialized_flag:
            self.logger.debug(
                f"SentenceTransformer '{self.model_name}' already initialized."
            )
            return
        self.logger.info(f"Initializing SentenceTransformer: {self.model_name}")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, SentenceTransformer, self.model_name
            )
            if not self.model:
                raise VectorStoreError(
                    f"Failed to load SentenceTransformer model {self.model_name}"
                )
            self.dimension = self.model.get_sentence_embedding_dimension()  # type: ignore
            self._initialized_flag = True
            self.logger.info(
                f"SentenceTransformer '{self.model_name}' loaded successfully.",
                parameters={"dimension": self.dimension},
            )
        except ImportError:
            self.logger.critical(
                "SentenceTransformers library not found. Install with: pip install sentence-transformers"
            )
            raise ConfigurationError(
                "SentenceTransformers library not installed.",
                config_key="embedding_provider_dependency",
            )
        except Exception as e:
            self.logger.critical(
                f"Failed to load SentenceTransformer model '{self.model_name}'.",
                exception=e,
                exc_info=True,
            )
            raise VectorStoreError(
                f"Model loading failed for {self.model_name}", cause=e
            )

    @detailed_log_function(LogCategory.VECTOR_STORE_EMBEDDING)
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=5),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
        before_sleep=tenacity.before_sleep_log(vs_embedding_logger, logging.WARNING),  # type: ignore
    )
    async def embed_texts(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        if not self._initialized_flag or not self.model:
            self.logger.warning(
                f"SentenceTransformer '{self.model_name}' not initialized. Attempting to initialize now."
            )
            await self.initialize()
            if not self._initialized_flag or not self.model:
                raise VectorStoreError(
                    f"SentenceTransformer model '{self.model_name}' could not be initialized for embedding."
                )

        if not texts:
            return []
        self.logger.debug(f"Embedding {len(texts)} texts using '{self.model_name}'.")
        start_time = time.perf_counter()
        loop = asyncio.get_event_loop()
        effective_batch_size = batch_size or self.config.get("embedding_batch_size", 64)

        try:
            embeddings_np_arrays: np.ndarray = await loop.run_in_executor(
                None,
                self.model.encode,
                texts,
                effective_batch_size,
                False,
                None,  # batch_size, show_progress_bar, output_value
                True,
                False,
                None,
                True,  # convert_to_numpy, normalize_embeddings (False, manual), convert_to_tensor, device
            )
        except Exception as e:
            self.logger.error(
                f"Error during SentenceTransformer encode for model '{self.model_name}'.",
                exception=e,
                exc_info=True,
            )
            raise VectorStoreError(
                f"Embedding generation failed with model {self.model_name}", cause=e
            )

        embeddings_normalized_lists: List[List[float]] = []
        for emb_np in embeddings_np_arrays:  # type: ignore
            norm = np.linalg.norm(emb_np)
            normalized_emb = (emb_np / norm if norm > 0 else emb_np).tolist()
            embeddings_normalized_lists.append(normalized_emb)

        duration = round(time.perf_counter() - start_time, 3)
        vs_perf_logger.performance_metric(
            "EmbedTexts_ST",
            duration,
            {
                "num_texts": len(texts),
                "batch_size": effective_batch_size,
                "model": self.model_name,
            },
        )
        self.logger.debug(
            f"Generated {len(embeddings_normalized_lists)} normalized embeddings in {duration}s."
        )
        return embeddings_normalized_lists


class VectorStore:
    GPU_SUPPORTED_INDEX_TYPES = [
        IndexType.FLAT,
        IndexType.IVF,
        IndexType.HNSW,
        IndexType.IVFPQ,
        IndexType.PQ,
    ]  # PQ can benefit too

    @detailed_log_function(LogCategory.VECTOR_STORE)
    def __init__(
        self,
        storage_path_str: str,
        embedding_provider: EmbeddingProviderVS,
        default_index_type: IndexType = IndexType.HNSW,
        enable_gpu_faiss: bool = False,
        *,
        document_index_path: str | None = None,
        entity_index_path: str | None = None,
        service_config: Optional[Dict[str, Any]] = None,
        cache_manager: Optional[Any] = None,
        metrics_exporter: Optional[Any] = None,
    ):
        vector_store_logger.info("=== VectorStore: Instance Creation START ===")
        self.config = service_config or {}
        self.cache_manager = cache_manager
        self.metrics = metrics_exporter
        self.storage_path = Path(storage_path_str)
        self.document_index_path = (
            Path(document_index_path)
            if document_index_path is not None
            else self.storage_path / "document_index.faiss"
        )
        self.entity_index_path = (
            Path(entity_index_path)
            if entity_index_path is not None
            else self.storage_path / "entity_index.faiss"
        )

        if not isinstance(embedding_provider, EmbeddingProviderVS):
            raise ConfigurationError(
                "Invalid EmbeddingProviderVS instance provided.",
                config_key="embedding_provider",
            )
        self.embedding_provider = embedding_provider
        if self.embedding_provider.dimension is None:
            msg = (
                "Injected embedding provider must be initialized and have a valid "
                "'dimension' attribute."
            )
            vector_store_logger.critical(msg)
            raise ConfigurationError(
                msg, config_key="embedding_provider.dimension (must be pre-initialized)"
            )
        self.dimension: int = self.embedding_provider.dimension

        self.state = VectorStoreState.UNINITIALIZED
        self.index_type = default_index_type
        self.enable_gpu = enable_gpu_faiss and faiss.get_num_gpus() > 0

        self.document_index: Optional[faiss.Index] = None
        self.entity_index: Optional[faiss.Index] = None

        self.ivf_training_threshold = int(
            self.config.get("ivf_training_threshold", 1024)
        )
        self.ivf_nlist_factor = int(self.config.get("ivf_nlist_factor", 8))
        self.document_index_requires_training = self.index_type in [
            IndexType.IVF,
            IndexType.IVFPQ,
            IndexType.PQ,
        ]
        self.document_index_trained = not self.document_index_requires_training
        self.pending_vectors_document: List[np.ndarray] = []
        self.pending_vector_ids_document: List[int] = []

        self.entity_index_requires_training = self.index_type in [
            IndexType.IVF,
            IndexType.IVFPQ,
            IndexType.PQ,
        ]
        self.entity_index_trained = not self.entity_index_requires_training
        self.pending_vectors_entity: List[np.ndarray] = []
        self.pending_vector_ids_entity: List[int] = []

        # Product Quantization configuration
        self.pq_m = int(self.config.get("pq_m", 8))
        self.pq_nbits = int(self.config.get("pq_nbits", 8))
        self.pq_quantizer = faiss.ProductQuantizer(
            self.dimension, self.pq_m, self.pq_nbits
        )
        self.pq_trained = False
        self._pq_training_data: List[np.ndarray] = []

        self.metadata_db_path = (
            self.storage_path / "vector_metadata.sqlite"
        )  # Changed extension
        cache_size = int(self.config.get("metadata_memory_cache_size", 10000))
        self.metadata_mem_cache: LRUCache[str, VectorMetadata] = LRUCache(
            maxsize=cache_size
        )

        self.vectorid_to_faissid_doc: Dict[str, int] = {}
        self.vectorid_to_faissid_entity: Dict[str, int] = {}
        self.faissid_to_vectorid_doc: Dict[int, str] = {}
        self.faissid_to_vectorid_entity: Dict[int, str] = {}

        self.stats = IndexStatistics()
        self.search_history_log: Deque[Dict[str, Any]] = deque(
            maxlen=self.config.get("search_history_max_len", 200)
        )

        self._async_lock = asyncio.Lock()
        self.metadata_lock = asyncio.Lock()
        self.faiss_lock = asyncio.Lock()

        self._optimization_task_internal: Optional[asyncio.Task] = None
        self._stop_background_tasks_event = (
            asyncio.Event()
        )  # One event for all background tasks
        self.optimization_request_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(
            maxsize=10
        )  # Max pending optimization jobs

        self._periodic_save_interval_sec = int(
            self.config.get("periodic_save_interval_sec", 300)
        )
        self._periodic_save_task_internal: Optional[asyncio.Task] = None

        vector_store_logger.info(
            "=== VectorStore: Instance Creation END (Uninitialized) ==="
        )

    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def initialize(self):
        if self.state not in [VectorStoreState.UNINITIALIZED, VectorStoreState.ERROR]:
            vector_store_logger.warning(
                f"VectorStore already initialized or in active state: {self.state.value}"
            )
            return

        self.state = VectorStoreState.INITIALIZING
        vector_store_logger.info("VectorStore initialization started.")
        self.storage_path.mkdir(
            parents=True, exist_ok=True
        )  # Ensure storage path exists
        try:
            if self.embedding_provider.dimension is None:  # Double check, should be set
                await self.embedding_provider.initialize()
                if self.embedding_provider.dimension is None:
                    raise VectorStoreError(
                        "Embedding provider dimension is None after initialization attempt."
                    )
            self.dimension = self.embedding_provider.dimension  # Re-affirm
            vector_store_logger.info(
                f"Using embedding provider '{self.embedding_provider.model_name}' dim: {self.dimension}."
            )

            await self._initialize_metadata_storage_async()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self._initialize_faiss_indexes_sync
            )  # Creates empty shells
            await self._load_all_existing_data_async()  # Loads from disk into shells

            # Recovery: if caches failed to load but index files exist, reload
            if not self.metadata_mem_cache and (
                self.document_index_path.exists() or self.entity_index_path.exists()
            ):
                vector_store_logger.warning(
                    "Caches empty after initial load; attempting recovery from disk."
                )
                await loop.run_in_executor(None, self._load_metadata_mem_cache_sync)
                await loop.run_in_executor(None, self._load_id_mapping_cache_sync)

            self._start_background_optimization_task()
            self._start_periodic_save_task()

            self.state = VectorStoreState.READY
            vector_store_logger.info(
                "VectorStore initialization complete and ready.",
                parameters=self._get_current_stats_summary(),
            )
        except Exception as e:
            self.state = VectorStoreState.ERROR
            vector_store_logger.critical(
                "VectorStore initialization failed critically.",
                exception=e,
                exc_info=True,
            )
            await self.close()  # Attempt graceful cleanup on failed init
            raise VectorStoreError("VectorStore failed to initialize.", cause=e)


    def _migrate_metadata_to_id_map_sync(self, conn: sqlite3.Connection) -> None:
        """Populate faiss_id_map table from existing metadata if missing."""
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM faiss_id_map")
            existing = cursor.fetchone()[0]
            if existing > 0:
                return

            doc_next = 0
            ent_next = 0
            cursor = conn.execute("SELECT vector_id, faiss_id FROM vector_metadata")
            rows = cursor.fetchall()
            for vector_id, faiss_id in rows:
                index_target = "entity" if vector_id.startswith("vec_ent") else "document"
                if index_target == "document":
                    if faiss_id is None or faiss_id < 0:
                        faiss_id = doc_next
                        doc_next += 1
                        conn.execute(
                            "UPDATE vector_metadata SET faiss_id = ? WHERE vector_id = ?",
                            (faiss_id, vector_id),
                        )
                    else:
                        doc_next = max(doc_next, int(faiss_id) + 1)
                else:
                    if faiss_id is None or faiss_id < 0:
                        faiss_id = ent_next
                        ent_next += 1
                        conn.execute(
                            "UPDATE vector_metadata SET faiss_id = ? WHERE vector_id = ?",
                            (faiss_id, vector_id),
                        )
                    else:
                        ent_next = max(ent_next, int(faiss_id) + 1)

                conn.execute(
                    "INSERT OR IGNORE INTO faiss_id_map (vector_id, index_target, faiss_id) VALUES (?,?,?)",
                    (vector_id, index_target, int(faiss_id)),
                )
            conn.commit()
        except sqlite3.Error as e:
            vs_index_logger.error("Failed migrating metadata to faiss_id_map", exception=e)

    def _initialize_faiss_indexes_sync(self):
        vs_index_logger.info(
            f"Initializing FAISS indexes (sync).",
            parameters={"type": self.index_type.value, "dim": self.dimension},
        )
        if self.dimension is None:
            raise ConfigurationError(
                "Dimension not set for FAISS index.", config_key="dimension"
            )

        def create_index(
            index_type_enum: IndexType, dim: int, purpose: str
        ) -> faiss.Index:
            vs_index_logger.debug(
                f"Creating FAISS index of type {index_type_enum.value} for {purpose} with dim {dim}"
            )
            base = self._create_base_index(index_type_enum, dim, purpose)
            return faiss.IndexIDMap2(base)

            # For IVF types, nlist needs to be determined.
            # A common heuristic: 4*sqrt(N) to 16*sqrt(N), where N is number of vectors.
            # If N is unknown at init, use a reasonable default or make it configurable.
            current_vector_count = (
                self.stats.total_vectors
                if self.stats.total_vectors > self.ivf_training_threshold
                else self.ivf_training_threshold
            )
            nlist = max(1, int(self.ivf_nlist_factor * np.sqrt(current_vector_count)))
            nlist = int(
                self.config.get(f"ivf_nlist_{purpose}", nlist)
            )  # Allow override
            vs_index_logger.debug(
                f"Calculated nlist={nlist} for {purpose} index (IVF type)."
            )

            quantizer = faiss.IndexFlatL2(dim)  # Common quantizer for IVF types
            if index_type_enum == IndexType.IVF:
                return faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

            # For PQ types, m must be a divisor of dim. nbits is usually 8.
            m_pq = dim  # Default to full dim if no good divisor
            possible_ms = [
                d
                for d in [
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    dim // 64,
                    dim // 32,
                    dim // 16,
                    dim // 8,
                    dim // 4,
                    dim // 2,
                ]
                if d > 0 and dim % d == 0
            ]
            if possible_ms:
                m_pq = possible_ms[
                    -1
                ]  # Prefer larger m (more subquantizers) from sensible options
            m_pq = int(self.config.get(f"pq_m_{purpose}", m_pq))
            if dim % m_pq != 0:
                vs_index_logger.error(
                    f"Dimension {dim} not divisible by m_pq {m_pq} for {index_type_enum.value} on {purpose}. This will fail."
                )
                raise ConfigurationError(
                    f"FAISS PQ m ({m_pq}) must be a divisor of dimension ({dim}).",
                    config_key=f"pq_m_{purpose}",
                )

            nbits_pq = int(self.config.get(f"pq_nbits_{purpose}", 8))
            if index_type_enum == IndexType.PQ:
                return faiss.IndexPQ(dim, m_pq, nbits_pq, faiss.METRIC_L2)
            if index_type_enum == IndexType.IVFPQ:
                return faiss.IndexIVFPQ(
                    quantizer, dim, nlist, m_pq, nbits_pq, faiss.METRIC_L2
                )

            raise ConfigurationError(
                f"Unsupported FAISS index type for creation: {index_type_enum.value}",
                config_key="index_type",
            )

        try:

                if self.enable_gpu:
                    try:
                        vs_index_logger.info("Attempting to use GPU for FAISS indexes.")
                        gpu_resource = faiss.StandardGpuResources()  # type: ignore[attr-defined]
                        if (
                            self.document_index
                            and self.index_type in self.GPU_SUPPORTED_INDEX_TYPES
                        ):
                            self.document_index = faiss.index_cpu_to_gpu(gpu_resource, 0, self.document_index)  # type: ignore[attr-defined]
                        if (
                            self.entity_index
                            and self.index_type in self.GPU_SUPPORTED_INDEX_TYPES
                        ):  # Assuming same type for entity index
                            self.entity_index = faiss.index_cpu_to_gpu(gpu_resource, 0, self.entity_index)  # type: ignore[attr-defined]
                        vs_index_logger.info(
                            "GPU acceleration successfully enabled for FAISS indexes."
                        )
                    except Exception as gpu_e:
                        vs_index_logger.warning(
                            "Failed to move FAISS indexes to GPU. Using CPU.",
                            exception=gpu_e,
                        )
                        self.enable_gpu = False  # Disable if failed

                vs_index_logger.info(
                    "FAISS indexes created in memory.",
                    parameters={
                        "doc_idx": type(self.document_index).__name__,
                        "ent_idx": type(self.entity_index).__name__,
                    },
                )
        except Exception as e:
            vs_index_logger.critical(
                "Failed to initialize FAISS indexes.", exception=e, exc_info=True
            )
            raise VectorStoreError(
                "FAISS index initialization critical failure.", cause=e
            )

    def _train_pq(self, vectors: np.ndarray) -> None:
        """Train the product quantizer on provided vectors if not already trained."""
        if self.pq_trained:
            return
        try:
            self.pq_quantizer.train(vectors)
            self.pq_trained = True
            vs_index_logger.info(
                "Product quantizer trained.",
                parameters={"samples": len(vectors)},
            )
        except Exception as e:
            vs_index_logger.error("PQ training failed", exception=e)

    async def _load_all_existing_data_async(self):
        vector_store_logger.trace("Loading existing data from storage (async).")
        await self._load_metadata_mem_cache_async()
        await self._load_id_mapping_cache_async()

        doc_vectors = self.document_index.ntotal if self.document_index else 0
        ent_vectors = self.entity_index.ntotal if self.entity_index else 0
        self.stats.total_vectors = doc_vectors + ent_vectors

        # Set trained flags based on loaded index content
        if self.document_index_requires_training:
            self.document_index_trained = (
                self.document_index.is_trained if self.document_index else False
            )
        if self.entity_index_requires_training:
            self.entity_index_trained = (
                self.entity_index.is_trained if self.entity_index else False
            )

        vector_store_logger.info(
            "Existing data loading process complete.",
            parameters=self._get_current_stats_summary(),
        )

    def _load_faiss_indexes_sync(self):
        """Synchronous part of loading FAISS indexes from disk."""
        doc_index_path = self.document_index_path
        entity_index_path = self.entity_index_path

        if doc_index_path.exists():
            try:
                if self.enable_gpu:
                    try:
                        gpu_resource = faiss.StandardGpuResources()  # type: ignore[attr-defined]
                        self.document_index = faiss.index_cpu_to_gpu(
                            gpu_resource, 0, self.document_index
                        )  # type: ignore[attr-defined]
                    except Exception as gpu_e:
                        vs_index_logger.warning(
                            f"Failed to move loaded document index to GPU: {gpu_e}. Using CPU."
                        )
                vectors = (
                    self.document_index.ntotal if self.document_index else 0
                )
                vs_index_logger.info(
                    "Document FAISS index loaded.",
                    parameters={"path": str(doc_index_path), "vectors": vectors},
                )
            except Exception as e:
                vs_index_logger.warning(
                    f"Failed to load document FAISS index from '{doc_index_path}'. Re-initializing empty.",
                    exception=e,
                )
                self._initialize_faiss_indexes_sync()  # Re-init if load fails to prevent None index
        else:
            vs_index_logger.info(
                f"No document FAISS index file found at '{doc_index_path}'. Starting with empty index."
            )

        entity_index_path = self.entity_index_path
        if entity_index_path.exists():
            try:
                if self.enable_gpu:
                    try:
                        gpu_resource = faiss.StandardGpuResources()  # type: ignore[attr-defined]
                        self.entity_index = faiss.index_cpu_to_gpu(
                            gpu_resource, 0, self.entity_index
                        )  # type: ignore[attr-defined]
                    except Exception as gpu_e:
                        vs_index_logger.warning(
                            f"Failed to move loaded entity index to GPU: {gpu_e}. Using CPU."
                        )
                ent_vectors = (
                    self.entity_index.ntotal if self.entity_index else 0
                )
                vs_index_logger.info(
                    "Entity FAISS index loaded.",
                    parameters={
                        "path": str(entity_index_path),
                        "vectors": ent_vectors,
                    },
                )
            except Exception as e:
                vs_index_logger.warning(
                    f"Failed to load entity FAISS index from '{entity_index_path}'. Re-initializing empty.",
                    exception=e,
                )
                # Re-init only entity index if document index was fine
                if not self.document_index:
                    self._initialize_faiss_indexes_sync()
                else:
                    # _initialize_faiss_indexes_sync updates instance attributes in-place
                    self._initialize_faiss_indexes_sync()

            vs_cache_logger.error("SQLite error loading metadata cache.", exception=e)
        except Exception as e:
            vs_cache_logger.error(
                "Unexpected error loading metadata cache.", exception=e, exc_info=True
            )

                )
        self.metadata_mem_cache = new_cache
        vs_cache_logger.info(
            "Metadata memory cache loaded.",
            parameters={"cached_items": len(self.metadata_mem_cache)},
        )

            return
        vs_index_logger.info("Starting background optimization task manager.")
        self._stop_background_tasks_event.clear()
        self._optimization_task_internal = asyncio.create_task(
            self._optimization_worker_async()
        )

    async def _optimization_worker_async(self):
        vs_index_logger.info("Async optimization worker task started.")
        while not self._stop_background_tasks_event.is_set():
            try:
                task_data = await asyncio.wait_for(
                    self.optimization_request_queue.get(), timeout=5.0
                )
                if task_data is None:  # Sentinel value to stop
                    self.optimization_request_queue.task_done()
                    break

                task_type = task_data.get("type", "UNKNOWN_OPTIMIZATION_TASK")
                vs_index_logger.info(f"Processing optimization task: {task_type}")
                async with self._async_lock:  # Ensure optimization logic has exclusive access
                    if task_type == "OPTIMIZE_INDEX_PERFORMANCE":
                        await self.optimize_performance(  # type: ignore[attr-defined]
                            full_reindex=task_data.get("full_reindex", False),
                            compact_storage=task_data.get("compact_storage", True),
                        )
                    elif task_type == "SAVE_INDEXES_NOW":  # Explicit save request
                        await self._save_faiss_indexes_async()
                    else:
                        vs_index_logger.warning(
                            f"Unknown optimization task type received: {task_type}"
                        )
                self.optimization_request_queue.task_done()
            except asyncio.TimeoutError:
                continue  # Loop to check _stop_background_tasks_event
            except asyncio.CancelledError:
                vs_index_logger.info("Optimization worker task was cancelled.")
                break
            except Exception as e:
                vs_index_logger.error(
                    "Error in async optimization worker.", exception=e, exc_info=True
                )
                await asyncio.sleep(30)  # Wait before retrying queue get after error
        vs_index_logger.info("Async optimization worker task stopped.")

    def _start_periodic_save_task(self):
        if (
            self._periodic_save_task_internal
            and not self._periodic_save_task_internal.done()
        ):
            vs_index_logger.warning("Periodic save task already running.")
            return
        vs_index_logger.info(
            f"Starting periodic save task (interval: {self._periodic_save_interval_sec}s)."
        )
        self._periodic_save_task_internal = asyncio.create_task(
            self._periodic_save_worker_async()
        )

    async def _periodic_save_worker_async(self):
        vs_index_logger.info("Periodic save worker task started.")
        while not self._stop_background_tasks_event.is_set():
            try:
                await asyncio.sleep(self._periodic_save_interval_sec)
                if self._stop_background_tasks_event.is_set():
                    break

                if self.state == VectorStoreState.READY:
                    vs_index_logger.info("Performing periodic save of FAISS indexes.")
                    await self.optimization_request_queue.put(
                        {"type": "SAVE_INDEXES_NOW"}
                    )  # Use queue for coordinated save
                else:
                    vs_index_logger.debug(
                        f"Skipping periodic save, store state is {self.state.value}."
                    )
            except asyncio.CancelledError:
                vs_index_logger.info("Periodic save task was cancelled.")
                break
            except Exception as e:
                vs_index_logger.error(
                    "Error in periodic save worker.", exception=e, exc_info=True
                )
                await asyncio.sleep(
                    self._periodic_save_interval_sec // 2
                )  # Shorter wait after error

    async def _save_faiss_indexes_async(self):
        """Save FAISS indexes using an asyncio lock for thread safety."""
        async with self.faiss_lock:
            loop = asyncio.get_event_loop()
            vs_index_logger.debug("Acquired lock for saving FAISS indexes.")
            try:
                await loop.run_in_executor(None, self._save_faiss_indexes_sync)
            finally:
                vs_index_logger.debug("Released lock after saving FAISS indexes.")

    def _save_faiss_indexes_sync(self):


    # --- FAISS ID Mapping Helpers ---
    def _get_next_faiss_id_sync(self, index_target: str) -> int:
        mapping = (
            self.vectorid_to_faissid_doc
            if index_target.lower() == "document"
            else self.vectorid_to_faissid_entity
        )
        return (max(mapping.values()) + 1) if mapping else 0

    def _store_id_mapping_sync(
        self, vector_id: str, faiss_id: int, index_target: str
    ) -> None:
        mapping_v2f = (
            self.vectorid_to_faissid_doc
            if index_target.lower() == "document"
            else self.vectorid_to_faissid_entity
        )
        mapping_f2v = (
            self.faissid_to_vectorid_doc
            if index_target.lower() == "document"
            else self.faissid_to_vectorid_entity
        )
        mapping_v2f[vector_id] = faiss_id
        mapping_f2v[faiss_id] = vector_id

    def _delete_id_mapping_sync(self, vector_id: str) -> None:
        if vector_id in self.vectorid_to_faissid_doc:
            fid = self.vectorid_to_faissid_doc.pop(vector_id)
            self.faissid_to_vectorid_doc.pop(fid, None)
        if vector_id in self.vectorid_to_faissid_entity:
            fid = self.vectorid_to_faissid_entity.pop(vector_id)
            self.faissid_to_vectorid_entity.pop(fid, None)

    def _get_faiss_id_for_vector_id_sync(
        self, vector_id: str, index_target: str
    ) -> Optional[int]:
        mapping = (
            self.vectorid_to_faissid_doc
            if index_target.lower() == "document"
            else self.vectorid_to_faissid_entity
        )
        if vector_id in mapping:
            return mapping[vector_id]
            )
            reverse_map[faiss_id] = vector_id
            return faiss_id
        return None

    def _get_vector_id_by_faiss_id_sync(
        self, faiss_id: int, index_target: str
    ) -> Optional[str]:
        reverse_map = (
            self.faissid_to_vectorid_doc
            if index_target.lower() == "document"
            else self.faissid_to_vectorid_entity
        )
        if faiss_id in reverse_map:
            return reverse_map[faiss_id]
        try:

    async def _delete_id_mapping_async(self, vector_id: str) -> None:
        self._delete_id_mapping_sync(vector_id)

    async def _get_vector_id_by_faiss_id_async(
        self, faiss_id: int, index_target: str
    ) -> Optional[str]:
        vector_id = await self.metadata_repo.get_vector_id_by_faiss_id(faiss_id, index_target)
        return vector_id

    async def _get_faiss_id_for_vector_id_async(
        self, vector_id: str, index_target: str
    ) -> Optional[int]:
        faiss_id = await self.metadata_repo.get_faiss_id_for_vector_id(vector_id, index_target)
        return faiss_id

    async def _get_metadata_async_from_db_or_cache(
        self, vector_id: str
    ) -> Optional[VectorMetadata]:
        """Retrieve metadata for a vector_id from cache or SQLite."""
        if vector_id in self.metadata_mem_cache:
            self.stats.cache_hits_session += 1
            meta = self.metadata_mem_cache[vector_id]
            meta.last_accessed_iso = datetime.now(timezone.utc).isoformat()
            meta.access_count += 1
            return meta

        self.stats.cache_misses_session += 1
        vs_cache_logger.debug(
            f"Metadata cache miss for vector_id '{vector_id}'. Fetching from DB."
        )
        loop = asyncio.get_event_loop()
        row_dict = await loop.run_in_executor(
            None, self._get_metadata_from_db_sync, vector_id
        )
        if row_dict:
            metadata = VectorMetadata.from_row_dict(row_dict)
            self.metadata_mem_cache[vector_id] = metadata
            return metadata
        return None

    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def add_vector_async(
        self,
        content_to_embed: str,
        document_id_ref: str,
        index_target: str = "document",  # "document" or "entity"
        vector_id_override: Optional[str] = None,
        vector_metadata_obj: Optional[
            VectorMetadata
        ] = None,  # Allow passing a pre-filled object
        **kwargs,  # For other metadata fields like source_file, tags, custom_metadata
    ) -> str:
        if self.state != VectorStoreState.READY:
            raise VectorStoreError(
                f"VectorStore not ready for adding vectors. Current state: {self.state.value}"
            )
        if not self.dimension:  # Should have been set by initialize
            raise VectorStoreError(
                "Vector dimension not set. Store may not be initialized."
            )

        vector_store_logger.info(
            f"Request to add vector.",
            parameters={"doc_id": document_id_ref, "index": index_target},
        )
        start_time_add = time.perf_counter()

        async with self._async_lock:  # Ensure atomicity of add + metadata store
            content_hash = hashlib.sha256(
                content_to_embed.encode("utf-8", "ignore")
            ).hexdigest()

            # Optional: Deduplication based on content hash
            if self.config.get("deduplicate_on_add", False):
                existing_vector_id = await self._find_by_content_hash_async(content_hash)
                if existing_vector_id:
                    vector_store_logger.info(
                        f"Duplicate content hash found. Returning existing vector_id '{existing_vector_id}'.",
                        parameters={"hash": content_hash, "doc_id": document_id_ref},
                    )
                    await self._update_access_stats_async(
                        existing_vector_id
                    )  # Update access for existing
                    return existing_vector_id

            # 1. Generate Embedding
            try:
                embeddings_list = await self.embedding_provider.embed_texts(
                    [content_to_embed]
                )
                if not embeddings_list or not embeddings_list[0]:
                    raise VectorStoreError(
                        "EmbeddingProvider returned no embedding for content."
                    )
                embedding_np = np.array(embeddings_list[0], dtype="float32").reshape(
                    1, -1
                )  # FAISS expects (1, D)
                # Train PQ with initial vectors
                if not self.pq_trained:
                    self._pq_training_data.append(embedding_np[0])
                    if len(self._pq_training_data) >= max(16, self.pq_m * 20):
                        train_np = np.vstack(self._pq_training_data).astype("float32")
                        self._train_pq(train_np)
                        self._pq_training_data.clear()
                pq_code_hex = ""
                if self.pq_trained:
                    code = self.pq_quantizer.compute_codes(embedding_np)
                    pq_code_hex = code.tobytes().hex()
            except Exception as embed_e:  # Catch errors from embedding provider
                raise VectorStoreError(
                    "Embedding generation failed.",
                    cause=embed_e,
                    details={"doc_id": document_id_ref},
                ) from embed_e

            # 2. Prepare FAISS index and metadata
            vector_norm_val = float(np.linalg.norm(embedding_np[0]))
            loop = asyncio.get_event_loop()

            # Determine target index and training status
            is_doc_index = index_target.lower() == "document"
            target_faiss_index = (
                self.document_index if is_doc_index else self.entity_index
            )
            requires_training = (
                self.document_index_requires_training
                if is_doc_index
                else self.entity_index_requires_training
            )
            is_trained_attr = (
                "document_index_trained" if is_doc_index else "entity_index_trained"
            )
            pending_vectors_list = (
                self.pending_vectors_document
                if is_doc_index
                else self.pending_vectors_entity
            )
            pending_ids_list = (
                self.pending_vector_ids_document
                if is_doc_index
                else self.pending_vector_ids_entity
            )

            if not target_faiss_index:
                raise VectorStoreError(
                    f"Target FAISS index '{index_target}' is not initialized."
                )

            faiss_id = await loop.run_in_executor(
                None, self._get_next_faiss_id_sync, index_target
            )

            # Handle IVF/PQ training if needed
            faiss_id = target_faiss_index.ntotal
            if requires_training and not getattr(self, is_trained_attr):
                pending_vectors_list.append(embedding_np[0])


            # 3. Create and Store Metadata
            # Use internal FAISS ID if not overridden. FAISS IDs are sequential 0 to ntotal-1.
            # This requires mapping FAISS ID to our application's vector_id.
            # A robust way is to assign our own UUIDs and store map: app_uuid -> faiss_internal_id
            # For simplicity here, if vector_id_override is not given, we use FAISS total count.
            # WARNING: This simple sequential ID from ntotal is fragile if removals happen without remapping!
            # It's better to use UUIDs for vector_id and manage the mapping to FAISS internal int64 IDs.
            # For now, we'll use a custom UUID-based vector_id for metadata.
            # FAISS itself will use its internal sequential IDs. The link is our responsibility.
            # Let's assume we use `add_with_ids` if we want to control FAISS IDs.
            # For `add`, FAISS assigns next available int64 ID. Retrieving this mapping from `add` is not direct.
            # Best practice: use IndexIDMap or IndexIDMap2 around the core index if you need custom string/UUID IDs in FAISS.
            # For now, our vector_id is application-level. We search by vector, get FAISS internal ID, then lookup app-level vector_id.

            vector_id = vector_id_override or self._generate_unique_vector_id(
                index_target, document_id_ref
            )

            if vector_metadata_obj:
                metadata_to_store = vector_metadata_obj
                metadata_to_store.vector_id = vector_id
                metadata_to_store.document_id = document_id_ref
                metadata_to_store.content_hash = content_hash
                metadata_to_store.content_preview = content_to_embed[:250]
                metadata_to_store.vector_norm = vector_norm_val
                metadata_to_store.dimension = self.dimension
                metadata_to_store.embedding_model = self.embedding_provider.model_name
                metadata_to_store.faiss_id = faiss_id
                if pq_code_hex:
                    metadata_to_store.custom_metadata["pq_code"] = pq_code_hex
            else:
                metadata_to_store = VectorMetadata(
                    faiss_id=faiss_id,
                    vector_id=vector_id,
                    document_id=document_id_ref,
                    content_hash=content_hash,
                    content_preview=content_to_embed[:250],
                    vector_norm=vector_norm_val,
                    dimension=self.dimension,
                )
                if pq_code_hex:
                    metadata_to_store.custom_metadata["pq_code"] = pq_code_hex

            await self._store_id_mapping_async(vector_id, faiss_id, index_target)
            self.stats.total_vectors = (
                self.document_index.ntotal if self.document_index else 0
            ) + (self.entity_index.ntotal if self.entity_index else 0)
            self.stats.total_adds_session += 1

            duration_add = round(time.perf_counter() - start_time_add, 3)
            self.stats.avg_add_time_sec_session = (
                (
                    self.stats.avg_add_time_sec_session
                    * (self.stats.total_adds_session - 1)
                    + duration_add
                )
                / self.stats.total_adds_session
                if self.stats.total_adds_session > 0
                else duration_add
            )

            vector_store_logger.info(
                "Vector added successfully.",
                parameters={
                    "vector_id": vector_id,
                    "doc_id": document_id_ref,
                    "index": index_target,
                    "duration_sec": duration_add,
                },
            )
            vs_perf_logger.performance_metric(
                f"AddVector_{index_target}",
                duration_add,
                {"content_len": len(content_to_embed)},
            )
            if self.metrics:
                self.metrics.observe_vector_add(duration_add)
            if self.cache_manager:
                try:
                    await self.cache_manager.set(
                        f"vec_meta:{vector_id}",
                        metadata_to_store.to_dict(),
                        ttl_seconds=self.config.get("metadata_cache_ttl", 3600),
                    )
                except Exception:
                    pass
            return vector_id

    def _generate_unique_vector_id(self, index_target: str, doc_ref: str) -> str:
        # Generate a more robust unique ID, not tied to FAISS internal count directly
        # This is the ID we will use in our metadata store.
        # The mapping to FAISS internal ID (0 to ntotal-1) happens at search/retrieval time.
        # OR, if using IndexIDMap, this would be the ID passed to FAISS.
        # For simplicity, assume we map during retrieval.
        ts_part = int(time.time() * 1000000)  # Microseconds for more uniqueness
        return f"vec_{index_target[:3]}_{hashlib.md5(doc_ref.encode()).hexdigest()[:8]}_{ts_part}"

    @detailed_log_function(LogCategory.VECTOR_STORE_SEARCH)
    async def search_similar_async(
        self,
        query_text: str,
        index_target: str = "document",
        top_k: int = 10,
        filter_metadata: Optional[
            Dict[str, Any]
        ] = None,  # For pre-filtering or post-filtering
        min_similarity_score: Optional[float] = None,
    ) -> List[SearchResult]:
        if self.state != VectorStoreState.READY:
            vs_search_logger.warning(
                f"VectorStore not ready for search. State: {self.state.value}"
            )
            return []
        if not self.dimension:
            raise VectorStoreError(
                "Vector dimension not set. Store may not be initialized properly."
            )

        async with self._async_lock:  # Lock during search to prevent index modification issues (if applicable)
            vs_search_logger.info(
                f"Performing similarity search.",
                parameters={
                    "query_preview": query_text[:50] + "...",
                    "index": index_target,
                    "k": top_k,
                },
            )
            start_time_search = time.perf_counter()

            target_faiss_index = (
                self.document_index
                if index_target.lower() == "document"
                else self.entity_index
            )
            is_trained_attr = (
                "document_index_trained"
                if index_target.lower() == "document"
                else "entity_index_trained"
            )

            if not target_faiss_index:
                raise VectorStoreError(
                    f"Target FAISS index '{index_target}' not initialized for search."
                )

            if getattr(
                self,
                (
                    "document_index_requires_training"
                    if index_target.lower() == "document"
                    else "entity_index_requires_training"
                ),
            ) and not getattr(self, is_trained_attr):
                vs_search_logger.warning(
                    f"{index_target} index requires training but is not trained. Search may be suboptimal or fail."
                )
                # Depending on index type, search might still work (e.g. on quantizer for IVF) but poorly.
                # For now, allow search but log warning. Could raise error for stricter behavior.

            # 1. Embed query
            try:
                query_embeddings_list = await self.embedding_provider.embed_texts(
                    [query_text]
                )
                if not query_embeddings_list or not query_embeddings_list[0]:
                    raise VectorStoreError(
                        "Failed to generate embedding for search query."
                    )
                query_embedding_np = np.array(
                    query_embeddings_list[0], dtype="float32"
                ).reshape(1, -1)
            except Exception as embed_e:
                raise VectorStoreError(
                    "Query embedding failed.", cause=embed_e
                ) from embed_e

            # 2. Perform FAISS search (synchronous, run in executor)
            loop = asyncio.get_event_loop()

            # HNSW specific search params
            if isinstance(target_faiss_index, faiss.IndexHNSW):
                target_faiss_index.hnsw.efSearch = int(
                    self.config.get("hnsw_efsearch", 64)
                )  # Set efSearch for HNSW quality
            elif isinstance(target_faiss_index, faiss.IndexIVF):  # IVF based indexes
                target_faiss_index.nprobe = int(
                    self.config.get("ivf_nprobe", 10)
                )  # Set nprobe for IVF quality

            try:
                distances_batch, indices_batch = await loop.run_in_executor(
                    None, target_faiss_index.search, query_embedding_np, top_k
                )
            except RuntimeError as faiss_e:  # Catch FAISS specific runtime errors
                raise VectorStoreError(
                    "FAISS search operation failed.", cause=faiss_e
                ) from faiss_e

            search_duration_faiss = (
                time.perf_counter() - start_time_search
            )  # Only FAISS search time

            # 3. Retrieve metadata and construct SearchResult objects
            results: List[SearchResult] = []
            cache_hits = 0

            faiss_indices = indices_batch[0]
            faiss_distances = distances_batch[0]

            for rank_idx, internal_faiss_id in enumerate(faiss_indices):
                if internal_faiss_id == -1:
                    continue  # padding value when fewer than k results

                metadata_obj = await self._get_metadata_by_faiss_internal_id_async(
                    int(internal_faiss_id), index_target
                )

                if metadata_obj:
                    if metadata_obj.vector_id in self.metadata_mem_cache:
                        cache_hits += 1

                    # L2 distance to similarity (common for normalized vectors: sim = 1 - dist^2/2, or 1 / (1+dist))
                    # For normalized vectors and L2 distance, cosine_similarity = 1 - (L2_distance^2 / 2)
                    # FAISS L2 distance is squared L2 distance.
                    similarity = (
                        1.0 - (faiss_distances[rank_idx] / 2.0)
                        if metadata_obj.vector_norm
                        else 0.0
                    )  # Assuming normalized vectors, L2 distance is squared
                    similarity = max(0.0, min(1.0, similarity))  # Clamp to [0,1]

                    if (
                        min_similarity_score is not None
                        and similarity < min_similarity_score
                    ):
                        continue

                    results.append(
                        SearchResult(
                            vector_id=metadata_obj.vector_id,
                            document_id=metadata_obj.document_id,
                            content_preview=metadata_obj.content_preview,
                            similarity_score=round(similarity, 4),
                            distance=round(float(faiss_distances[rank_idx]), 4),
                            metadata=metadata_obj,
                            search_time_sec=round(
                                search_duration_faiss, 5
                            ),  # Time for this specific result is part of batch time
                            index_type_used=self.index_type.value,
                            rank=rank_idx + 1,
                        )
                    )
                    await self._update_access_stats_async(
                        metadata_obj.vector_id, metadata_obj.last_accessed_iso
                    )
                else:
                    vs_search_logger.warning(
                        f"Metadata not found for FAISS internal_id {internal_faiss_id} in {index_target} index."
                    )

            # Apply post-filtering if any
            if filter_metadata:
                results = self._apply_filters_to_results(results, filter_metadata)

            total_search_pipeline_duration = round(
                time.perf_counter() - start_time_search, 3
            )
            self.stats.total_searches_session += 1
            self.stats.avg_search_time_sec_session = (
                (
                    self.stats.avg_search_time_sec_session
                    * (self.stats.total_searches_session - 1)
                    + total_search_pipeline_duration
                )
                / self.stats.total_searches_session
                if self.stats.total_searches_session > 0
                else total_search_pipeline_duration
            )

            if top_k > 0:
                self.stats.cache_hit_rate_session = (
                    (
                        self.stats.cache_hit_rate_session
                        * (self.stats.total_searches_session - 1)
                        + (cache_hits / top_k)
                    )
                    / self.stats.total_searches_session
                    if self.stats.total_searches_session > 0
                    else (cache_hits / top_k)
                )
            self.stats.cache_hits_session += cache_hits
            self.stats.cache_misses_session += top_k - cache_hits  # Approximate misses

            self.search_history_log.append(
                {
                    "query": query_text[:100],
                    "num_results": len(results),
                    "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                }
            )
            vs_search_logger.info(
                f"Search returned {len(results)} results.",
                parameters={
                    "duration_total_sec": total_search_pipeline_duration,
                    "faiss_time_sec": search_duration_faiss,
                },
            )
            if self.metrics:
                self.metrics.observe_vector_search(total_search_pipeline_duration)
            return results

    async def _get_metadata_by_faiss_internal_id_async(
        self, faiss_id: int, index_target: str
    ) -> Optional[VectorMetadata]:

    def _get_metadata_from_db_sync(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Synchronously fetches a single metadata record from SQLite."""
        try:
            with sqlite3.connect(
                self.metadata_db_path, timeout=5
            ) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM vector_metadata WHERE vector_id = ?", (vector_id,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            vs_cache_logger.error(
                f"SQLite error fetching metadata for '{vector_id}'.", exception=e
            )
            return None

    def _apply_filters_to_results(
        self, results: List[SearchResult], filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """Applies post-search filters to results based on metadata criteria."""
        if not filters:
            return results
        vs_search_logger.debug(
            f"Applying {len(filters)} post-search filters to {len(results)} results."
        )

        filtered_results: List[SearchResult] = []
        for res in results:
            match_all_filters = True
            for key, expected_value in filters.items():
                # Check against VectorMetadata fields directly or custom_metadata
                actual_value = None
                if hasattr(res.metadata, key):
                    actual_value = getattr(res.metadata, key)
                elif key in res.metadata.custom_metadata:
                    actual_value = res.metadata.custom_metadata[key]

                if isinstance(
                    expected_value, list
                ):  # If filter value is a list, check if actual_value is in it
                    if actual_value not in expected_value:
                        match_all_filters = False
                        break
                elif actual_value != expected_value:
                    match_all_filters = False
                    break

            if match_all_filters:
                filtered_results.append(res)

        vs_search_logger.debug(
            f"Filtering reduced results from {len(results)} to {len(filtered_results)}."
        )
        return filtered_results

    async def _update_access_stats_async(
        self, vector_id: str, current_last_accessed_iso: str
    ):
        """Updates last_accessed and access_count for a vector in metadata store."""
        new_last_accessed_iso = datetime.now(timezone.utc).isoformat()
        # Avoid DB write if accessed very recently (e.g., within last minute) unless count changes significantly
        # For simplicity, always update for now.


    @detailed_log_function(LogCategory.VECTOR_STORE_DB)
    async def update_vector_metadata_async(
        self, vector_id: str, metadata_updates: Dict[str, Any]
    ):
        """Updates specific fields in a vector's metadata."""
        if not vector_id or not metadata_updates:
            vector_store_logger.warning(
                "Update_vector_metadata_async called with empty vector_id or updates."
            )
            return

        async with self.metadata_lock:
            vs_cache_logger.debug(
                f"Updating metadata for vector_id '{vector_id}'.",
                parameters=metadata_updates,
            )

            # Update in-memory cache first
            if vector_id in self.metadata_mem_cache:
                cached_meta = self.metadata_mem_cache[vector_id]
                for key, value in metadata_updates.items():
                    if hasattr(cached_meta, key):
                        setattr(cached_meta, key, value)
                    else:
                        cached_meta.custom_metadata[
                            key
                        ] = value  # Put in custom_metadata if not a direct field
                cached_meta.last_accessed_iso = datetime.now(
                    timezone.utc
                ).isoformat()  # Also treat as access
                cached_meta.access_count += 1

            # Persist to DB
            await self.metadata_repo.update_metadata_fields(vector_id, metadata_updates)
            vector_store_logger.info(f"Metadata updated for vector_id '{vector_id}'.")


    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def delete_vector_async(self, vector_id: str, index_target: str = "document"):
        async with self._async_lock:
    def _update_disk_usage_stats(self):
        """Updates statistics about disk usage for indexes and metadata DB."""
        try:
            doc_idx_path = self.document_index_path
            ent_idx_path = self.entity_index_path
            meta_db_path = self.metadata_db_path

            doc_idx_size = doc_idx_path.stat().st_size if doc_idx_path.exists() else 0
            ent_idx_size = ent_idx_path.stat().st_size if ent_idx_path.exists() else 0
            meta_db_size = meta_db_path.stat().st_size if meta_db_path.exists() else 0

            self.stats.index_disk_size_mb = round(
                (doc_idx_size + ent_idx_size) / (1024 * 1024), 2
            )
            self.stats.metadata_db_size_mb = round(meta_db_size / (1024 * 1024), 2)
            self.stats.disk_usage_mb = (
                self.stats.index_disk_size_mb + self.stats.metadata_db_size_mb
            )
        except Exception as e:
            vs_perf_logger.warning(f"Could not update disk usage stats: {str(e)}")

    def _get_current_stats_summary(self) -> Dict[str, Any]:
        """Return a lightweight summary of current index statistics."""
        return {
            "total_vectors": self.stats.total_vectors,
            "index_disk_size_mb": self.stats.index_disk_size_mb,
            "metadata_db_size_mb": self.stats.metadata_db_size_mb,
            "optimization_runs": self.stats.optimization_runs_count,
        }

    # --- Service Lifecycle Methods ---
    async def close(self):
        vector_store_logger.info("Shutting down VectorStore...")
        if self.state == VectorStoreState.SHUTDOWN:
            vector_store_logger.info("VectorStore already shut down.")
            return
        self.state = VectorStoreState.SHUTDOWN

        self._stop_background_tasks_event.set()  # Signal all background tasks to stop

        # Stop periodic save task
        if self._periodic_save_task_internal:
            self._periodic_save_task_internal.cancel()
            try:
                await self._periodic_save_task_internal
            except asyncio.CancelledError:
                vector_store_logger.debug("Periodic save task cancelled.")
            except Exception as e:
                vector_store_logger.error(
                    "Error stopping periodic save task.", exception=e
                )
            self._periodic_save_task_internal = None

        # Stop optimization worker task
        if self._optimization_task_internal:
            try:
                await self.optimization_request_queue.put(None)  # Send sentinel
            except Exception as q_e:
                vector_store_logger.warning(
                    f"Error putting sentinel on optimization queue: {q_e}"
                )  # Queue might be full or closed

            # Give it a moment to process sentinel
            try:
                await asyncio.wait_for(self._optimization_task_internal, timeout=7.0)
            except asyncio.TimeoutError:
                vector_store_logger.warning(
                    "Optimization worker task did not finish in time. Cancelling."
                )
                self._optimization_task_internal.cancel()
                try:
                    await self._optimization_task_internal
                except asyncio.CancelledError:
                    vector_store_logger.debug("Optimization worker task cancelled.")
            except Exception as e:
                vector_store_logger.error(
                    "Error stopping optimization worker task.", exception=e
                )
            self._optimization_task_internal = None

        # Perform final save of FAISS indexes
        vector_store_logger.info(
            "Performing final save of FAISS indexes before shutdown."
        )
        await self._save_faiss_indexes_async()

        vector_store_logger.info("VectorStore shutdown complete.")

    async def health_check(self) -> Dict[str, Any]:
        """Provides a comprehensive health status of the VectorStore."""
        vs_search_logger.debug("Performing VectorStore health check.")
        is_ready = self.state == VectorStoreState.READY
        status_label = "healthy" if is_ready else self.state.value

        # Try a small, quick test search if ready (optional, can be intensive)
        test_search_ok = None  # Placeholder for optional search test
        # if is_ready and self.stats.total_vectors > 0:
        #     try:
        #         test_query_vec = np.random.rand(1, self.dimension).astype('float32')
        #         loop = asyncio.get_event_loop()
        #         if self.document_index:
        #              await loop.run_in_executor(None, self.document_index.search, test_query_vec, 1)
        #         test_search_ok = True
        #     except Exception:
        #         test_search_ok = False
        #         status_label = "degraded_search_test_failed"

        self._update_disk_usage_stats()  # Update disk usage for health check

        return {
            "service_name": "VectorStore",
            "status": status_label,
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "state": self.state.value,
            "embedding_provider": {
                "model_name": self.embedding_provider.model_name,
                "dimension": self.dimension,
                "initialized": self.embedding_provider._initialized_flag
                if hasattr(self.embedding_provider, "_initialized_flag")
                else "unknown",
            },
            "index_config": {
                "type": self.index_type.value,
                "gpu_enabled": self.enable_gpu,
                "doc_index_vectors": self.document_index.ntotal
                if self.document_index
                else 0,
                "doc_index_trained": self.document_index.is_trained
                if self.document_index and hasattr(self.document_index, "is_trained")
                else self.document_index_trained,
                "entity_index_vectors": self.entity_index.ntotal
                if self.entity_index
                else 0,
                "entity_index_trained": self.entity_index.is_trained
                if self.entity_index and hasattr(self.entity_index, "is_trained")
                else self.entity_index_trained,
            },
            "statistics": asdict(self.stats),  # Convert dataclass to dict
            "background_tasks": {
                "optimization_worker_running": bool(
                    self._optimization_task_internal
                    and not self._optimization_task_internal.done()
                ),
                "periodic_save_running": bool(
                    self._periodic_save_task_internal
                    and not self._periodic_save_task_internal.done()
                ),
            },
            # "test_search_status": test_search_ok
        }

    async def get_service_status(
        self,
    ) -> Dict[str, Any]:  # For service container interface
        """Simplified status for service discovery, more detailed in health_check."""
        return {
            "name": "VectorStore",
            "status": "ready"
            if self.state == VectorStoreState.READY
            else self.state.value,
            "total_vectors": self.stats.total_vectors,
            "index_type": self.index_type.value,
            "last_updated_iso": datetime.now(
                timezone.utc
            ).isoformat(),  # Generic timestamp
        }

    async def optimize_performance(
        self, full_reindex: bool = False, compact_storage: bool = True
    ) -> Dict[str, Any]:
        vs_index_logger.info(
            "Starting performance optimization.",
            parameters={
                "full_reindex": full_reindex,
                "compact_storage": compact_storage,
            },
        )
        loop = asyncio.get_event_loop()
        if compact_storage:
            await loop.run_in_executor(None, self._compact_index_sync, True)
            await loop.run_in_executor(None, self._compact_index_sync, False)
        now = datetime.now(timezone.utc)
        self.stats.last_optimization_at_iso = now.isoformat()
        self.stats.optimization_runs_count += 1
        return {"optimized": True, "timestamp": self.stats.last_optimization_at_iso}

    def _compact_index_sync(self, is_doc_index: bool) -> None:
        index_target = "document" if is_doc_index else "entity"
        index = self.document_index if is_doc_index else self.entity_index
        mapping = (
            self.vectorid_to_faissid_doc
            if is_doc_index
            else self.vectorid_to_faissid_entity
        )
        if not index or not mapping:
            return
        ids = np.array(list(mapping.values()), dtype=np.int64)
        if len(ids) == 0:
            return
        vectors = np.vstack([index.reconstruct(int(fid)) for fid in ids])
        new_index = faiss.IndexIDMap2(
            self._create_base_index(self.index_type, self.dimension, index_target)
        )
        if (
            self.document_index_requires_training
            if is_doc_index
            else self.entity_index_requires_training
        ) and not getattr(
            self, "document_index_trained" if is_doc_index else "entity_index_trained"
        ):
            new_index.train(vectors)
            if is_doc_index:
                self.document_index_trained = True
            else:
                self.entity_index_trained = True
        new_index.add_with_ids(vectors.astype("float32"), ids)
        if self.enable_gpu:
            try:
                gpu_resource = faiss.StandardGpuResources()  # type: ignore[attr-defined]
                new_index = faiss.index_cpu_to_gpu(gpu_resource, 0, new_index)  # type: ignore[attr-defined]
            except Exception as e:
                vs_index_logger.warning(
                    "Failed to move compacted index to GPU.", exception=e
                )
        if is_doc_index:
            self.document_index = new_index
        else:
            self.entity_index = new_index

    def _create_base_index(
        self, index_type_enum: IndexType, dim: int, purpose: str
    ) -> faiss.Index:
        """Helper to create a base FAISS index without IDMap wrapper."""
        if index_type_enum == IndexType.FLAT:
            return faiss.IndexFlatL2(dim)
        if index_type_enum == IndexType.HNSW:
            idx = faiss.IndexHNSWFlat(
                dim, int(self.config.get(f"hnsw_m_{purpose}", 32)), faiss.METRIC_L2
            )
            idx.hnsw.efConstruction = int(
                self.config.get(f"hnsw_efconstruction_{purpose}", 40)
            )
            return idx
        current_vector_count = (
            self.stats.total_vectors
            if self.stats.total_vectors > self.ivf_training_threshold
            else self.ivf_training_threshold
        )
        nlist = max(1, int(self.ivf_nlist_factor * np.sqrt(current_vector_count)))
        nlist = int(self.config.get(f"ivf_nlist_{purpose}", nlist))
        quantizer = faiss.IndexFlatL2(dim)
        if index_type_enum == IndexType.IVF:
            return faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        m_pq = dim
        possible_ms = [
            d
            for d in [
                1,
                2,
                4,
                8,
                16,
                32,
                64,
                dim // 64,
                dim // 32,
                dim // 16,
                dim // 8,
                dim // 4,
                dim // 2,
            ]
            if d > 0 and dim % d == 0
        ]
        if possible_ms:
            m_pq = possible_ms[-1]
        m_pq = int(self.config.get(f"pq_m_{purpose}", m_pq))
        nbits_pq = int(self.config.get(f"pq_nbits_{purpose}", 8))
        if index_type_enum == IndexType.PQ:
            return faiss.IndexPQ(dim, m_pq, nbits_pq, faiss.METRIC_L2)
        if index_type_enum == IndexType.IVFPQ:
            return faiss.IndexIVFPQ(
                quantizer, dim, nlist, m_pq, nbits_pq, faiss.METRIC_L2
            )
        raise ConfigurationError(
            f"Unsupported FAISS index type {index_type_enum.value}"
        )

    # The `optimize_performance` method is already defined based on your previous request.


# Factory function at the end of vector_store.py
def create_vector_store(
    service_container: Any, service_config_override: Optional[Dict[str, Any]] = None
) -> VectorStore:
    """
    Factory function to create a VectorStore instance.
    IMPORTANT: The returned instance needs `await instance.initialize()` to be called by the service container.
    The EmbeddingProvider dependency should be resolved and initialized by the service container
    and then passed to this factory or directly to VectorStore constructor.
    """
    cfg_source = (
        service_config_override
        if service_config_override is not None
        else service_container
    )
    vs_cfg = cfg_source.get_config("vector_store_config", {})  # Get config slice

    # Resolve EmbeddingProviderVS dependency from service_container
    # This is crucial: embedding_provider must be INITIALIZED before VectorStore uses its dimension.
    try:
        # Assume service_container has a way to get an *initialized* embedding provider
        embedding_provider_instance: EmbeddingProviderVS = (
            service_container.get_service("embedding_provider")
        )
        if (
            not hasattr(embedding_provider_instance, "dimension")
            or embedding_provider_instance.dimension is None
        ):
            # This indicates the service container didn't provide a fully initialized provider.
            # The factory could try to initialize it, but it's better if the container manages service lifecycles.
            vector_store_logger.warning(
                "Embedding provider from service container does not have dimension. "
                "Ensure it's initialized before VectorStore creation."
            )
            # As a fallback, if this factory *must* work, it would need to know how to create and init one:
            # embedding_model_name = vs_cfg.get("embedding_provider", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
            # embedding_provider_instance = SentenceTransformerEmbeddingProvider(model_name=embedding_model_name, service_config=vs_cfg.get("embedding_provider",{}))
            # This is problematic because initialize is async. The factory is sync.
            # Best pattern: service container handles async init of provider THEN calls this sync factory.
            # If factory MUST be async: `async def create_vector_store_async(...)`
            raise ConfigurationError(
                "EmbeddingProvider from service_container is not initialized (no dimension)."
            )

    except (
        Exception
    ) as e:  # Catch if service "embedding_provider" is not found or other issues
        vector_store_logger.critical(
            f"Failed to get/initialize EmbeddingProvider from service_container: {e}. VectorStore cannot be created.",
            exc_info=True,
        )
        raise ConfigurationError(
            "Failed to obtain a valid EmbeddingProvider for VectorStore.",
            cause=e,
            config_key="embedding_provider_service",
        )

    vs_instance = VectorStore(
        storage_path_str=str(
            vs_cfg.get("STORAGE_PATH", "./storage/vector_store_data_default")
        ),
        embedding_provider=embedding_provider_instance,  # Pass the initialized provider
        default_index_type=IndexType(vs_cfg.get("DEFAULT_INDEX_TYPE", "HNSW").upper()),
        enable_gpu_faiss=bool(vs_cfg.get("ENABLE_GPU_FAISS", False)),
        document_index_path=vs_cfg.get("DOCUMENT_INDEX_PATH"),
        entity_index_path=vs_cfg.get("ENTITY_INDEX_PATH"),
        service_config=vs_cfg,
        cache_manager=getattr(_get_service_sync(service_container, "persistence_manager"), "cache_manager", None),
        metrics_exporter=_get_service_sync(service_container, "metrics_exporter"),
    )
    return vs_instance
