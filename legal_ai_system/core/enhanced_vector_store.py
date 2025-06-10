"""
Enhanced Vector Store - Consolidated with DETAILED Logging
=========================================================
Unified vector storage combining the best features from ultimate_vector_store.py,
vector_store_enhanced.py, and optimized_vector_store.py with comprehensive
detailed logging of every operation, every search, every data change.
"""

import json
import time
import faiss
from faiss import StandardGpuResources, index_cpu_to_gpu
import numpy as np
import threading
import asyncio
import asyncpg
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from queue import Queue
from .enhanced_persistence import ConnectionPool

# Import detailed logging system
from .detailed_logging import (
    get_detailed_logger,
    LogCategory,
    detailed_log_function,
)
from .enhanced_persistence import ConnectionPool, TransactionManager

# Initialize detailed loggers for different vector store operations
vector_logger = get_detailed_logger("Enhanced_Vector_Store", LogCategory.VECTOR_STORE)
search_logger = get_detailed_logger("Vector_Search", LogCategory.VECTOR_STORE)
embedding_logger = get_detailed_logger("Embedding_Operations", LogCategory.VECTOR_STORE)
index_logger = get_detailed_logger("Index_Management", LogCategory.VECTOR_STORE)
cache_logger = get_detailed_logger("Vector_Cache", LogCategory.VECTOR_STORE)
performance_logger = get_detailed_logger("Vector_Performance", LogCategory.PERFORMANCE)


class VectorStoreState(Enum):
    """Vector store operational states"""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    INDEXING = "indexing"
    SEARCHING = "searching"
    OPTIMIZING = "optimizing"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class IndexType(Enum):
    """Supported FAISS index types with performance characteristics"""

    FLAT = "IndexFlatL2"  # Exact search, high accuracy
    IVF = "IndexIVFFlat"  # Approximate search, good speed/accuracy
    HNSW = "IndexHNSWFlat"  # Hierarchical search, excellent speed
    PQ = "IndexPQ"  # Product quantization, memory efficient
    IVFPQ = "IndexIVFPQ"  # Combined IVF+PQ, balanced


@dataclass
class VectorMetadata:
    """Comprehensive metadata for vector entries"""

    vector_id: str
    document_id: str
    content_hash: str
    content_preview: str
    vector_norm: float
    dimension: int
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    source_file: Optional[str] = None
    document_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    embedding_model: str = "unknown"
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Comprehensive search result with detailed metadata"""

    vector_id: str
    document_id: str
    content_preview: str
    similarity_score: float
    distance: float
    metadata: VectorMetadata
    search_time: float
    index_used: str
    rank: int


@dataclass
class IndexStatistics:
    """Detailed index performance statistics"""

    total_vectors: int = 0
    index_size_mb: float = 0.0
    average_search_time: float = 0.0
    cache_hit_rate: float = 0.0
    total_searches: int = 0
    last_optimization: Optional[datetime] = None
    optimization_count: int = 0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0


class EmbeddingProvider:
    """Enhanced embedding provider with multiple model support and detailed logging"""

    @detailed_log_function(LogCategory.VECTOR_STORE)
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedding provider with detailed logging"""
        embedding_logger.info(f"Initializing embedding provider: {model_name}")

        self.model_name = model_name
        self.model = None
        self.dimension = None
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_embeddings = 0
        self.total_embedding_time = 0.0

        # Try multiple embedding approaches
        self._initialize_embedding_model()

        embedding_logger.info(
            "Embedding provider initialized",
            parameters={
                "model_name": self.model_name,
                "dimension": self.dimension,
                "cache_enabled": True,
            },
        )

    @detailed_log_function(LogCategory.VECTOR_STORE)
    def _initialize_embedding_model(self):
        """Initialize embedding model with fallback options"""
        embedding_logger.trace("Initializing embedding model with fallbacks")

        try:
            # Try SentenceTransformers first
            embedding_logger.trace("Attempting SentenceTransformers initialization")
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()

            embedding_logger.info(
                "SentenceTransformers model loaded successfully",
                parameters={"model": self.model_name, "dimension": self.dimension},
            )
            return

        except ImportError:
            embedding_logger.warning(
                "SentenceTransformers not available, trying Ollama"
            )

        try:
            # Try Ollama as fallback
            embedding_logger.trace("Attempting Ollama initialization")
            import requests

            # Test Ollama connection
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                self.model = "ollama"
                self.dimension = 384  # Default embedding dimension
                embedding_logger.info("Ollama embedding provider initialized")
                return

        except Exception as e:
            embedding_logger.warning("Ollama not available", exception=e)

        # Final fallback - simple TF-IDF style embeddings
        embedding_logger.warning("Using fallback embedding method")
        self.model = "fallback"
        self.dimension = 128

        embedding_logger.info("Fallback embedding provider initialized")

    @detailed_log_function(LogCategory.VECTOR_STORE)
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings with comprehensive logging and caching"""
        embedding_logger.trace(f"Generating embedding for text (length: {len(text)})")

        start_time = time.time()

        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            self.cache_hits += 1
            embedding = self.embedding_cache[text_hash]
            cache_time = time.time() - start_time

            embedding_logger.trace(
                "Embedding retrieved from cache",
                parameters={
                    "text_hash": text_hash[:8],
                    "cache_time": cache_time,
                    "cache_hit_rate": self.cache_hits
                    / (self.cache_hits + self.cache_misses),
                },
            )

            return embedding

        self.cache_misses += 1

        try:
            if isinstance(self.model, object) and hasattr(self.model, "encode"):
                # SentenceTransformers
                embedding_logger.trace("Using SentenceTransformers for embedding")
                embedding = np.asarray(self.model.encode([text])[0])

            elif self.model == "ollama":
                # Ollama API
                embedding_logger.trace("Using Ollama API for embedding")
                embedding = self._get_ollama_embedding(text)

            else:
                # Fallback method
                embedding_logger.trace("Using fallback embedding method")
                embedding = self._get_fallback_embedding(text)

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

            # Cache the result
            self.embedding_cache[text_hash] = embedding

            # Limit cache size
            if len(self.embedding_cache) > 10000:
                # Remove oldest 20% of entries
                cache_items = list(self.embedding_cache.items())
                self.embedding_cache = dict(cache_items[-8000:])
                cache_logger.info(
                    "Embedding cache pruned",
                    parameters={"new_size": len(self.embedding_cache)},
                )

            embedding_time = time.time() - start_time
            self.total_embeddings += 1
            self.total_embedding_time += embedding_time

            embedding_logger.info(
                "Embedding generated successfully",
                parameters={
                    "text_length": len(text),
                    "embedding_dimension": len(embedding),
                    "embedding_time": embedding_time,
                    "cache_hit_rate": self.cache_hits
                    / (self.cache_hits + self.cache_misses),
                    "total_embeddings": self.total_embeddings,
                },
            )

            return embedding

        except Exception as e:
            embedding_logger.error(
                "Embedding generation failed",
                exception=e,
                parameters={
                    "text_length": len(text),
                    "model_type": str(type(self.model)),
                },
            )
            raise

    @detailed_log_function(LogCategory.VECTOR_STORE)
    def _get_ollama_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama API"""
        import requests

        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=30,
        )

        if response.status_code == 200:
            return np.array(response.json()["embedding"])
        else:
            raise RuntimeError(f"Ollama API error: {response.status_code}")

    @detailed_log_function(LogCategory.VECTOR_STORE)
    def _get_fallback_embedding(self, text: str) -> np.ndarray:
        """Generate simple fallback embedding"""
        # Simple character-based embedding for fallback
        embedding = np.zeros((self.dimension,))

        for i, char in enumerate(text[: self.dimension]):
            embedding[i] = ord(char) / 255.0

        return embedding


class EnhancedVectorStore:
    """
    Comprehensive vector store with detailed logging, multiple index types,
    intelligent caching, and performance optimization.
    """

    @detailed_log_function(LogCategory.VECTOR_STORE)
    def __init__(
        self,
        storage_path: str = "./storage/vectors",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_type: IndexType = IndexType.HNSW,
        enable_gpu: bool = False,
        index_params: Optional[Dict[str, Any]] = None,
        document_index_path: str | None = None,
        entity_index_path: str | None = None,

    ):
        """Initialize enhanced vector store with comprehensive configuration"""
        vector_logger.info("=== INITIALIZING ENHANCED VECTOR STORE ===")

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

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

        self.state = VectorStoreState.UNINITIALIZED
        self.index_type = index_type
        self.enable_gpu = enable_gpu
        # Core components
        self.embedding_provider = EmbeddingProvider(embedding_model)
        self.dimension = self.embedding_provider.dimension

        # FAISS indexes
        self.document_index = None
        self.entity_index = None

        # Metadata storage
        self.metadata_db_path = self.storage_path / "metadata.db"
        self.metadata_cache = {}
        # Mapping between FAISS index positions and vector IDs
        self.index_id_map: Dict[int, str] = {}
        self.vector_id_to_index: Dict[str, int] = {}

        # Performance tracking
        self.statistics = IndexStatistics()
        self.search_history = []
        self.optimization_queue = Queue()

        # Thread safety
        self._lock = threading.RLock()

        # Persistence integration
        self.pool = connection_pool
        self.transaction_manager = (
            TransactionManager(connection_pool) if connection_pool else None
        )

        # Async lock for async methods
        self._async_lock = asyncio.Lock()

        self.state = VectorStoreState.UNINITIALIZED

        vector_logger.info(
            "Enhanced vector store instance created",
            parameters={
                "storage_path": str(self.storage_path),
                "dimension": self.dimension,
                "index_type": self.index_type.value,
                "gpu_enabled": self.enable_gpu,
                "total_vectors": self.statistics.total_vectors,
            },
        )

    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def initialize(self) -> None:
        """Asynchronously initialize storage and indexes."""
        if self.state != VectorStoreState.UNINITIALIZED:
            return

        self.state = VectorStoreState.INITIALIZING
        await self._initialize_storage()
        self._initialize_indexes()

        self.state = VectorStoreState.READY
        vector_logger.info("Enhanced vector store initialization complete")

    async def close(self) -> None:
        """Close connection pool if managed externally."""
        if self.pool:
            await self.pool.close()

    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def _initialize_storage(self) -> None:
        """Initialize PostgreSQL storage for metadata using the connection pool."""
        if not self.pool:
            raise RuntimeError("ConnectionPool is required for storage initialization")

        vector_logger.trace("Initializing metadata storage")

        async with self.pool.get_pg_connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_metadata (
                    vector_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    content_preview TEXT,
                    vector_norm REAL,
                    dimension INTEGER,
                    created_at TIMESTAMPTZ,
                    last_accessed TIMESTAMPTZ,
                    access_count INTEGER DEFAULT 0,
                    source_file TEXT,
                    document_type TEXT,
                    tags JSONB,
                    confidence_score REAL DEFAULT 1.0,
                    embedding_model TEXT,
                    custom_metadata JSONB
                )
            """,
            )

            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_document_id ON vector_metadata(document_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_content_hash ON vector_metadata(content_hash)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_last_accessed ON vector_metadata(last_accessed)"
            )

        vector_logger.info("Metadata storage initialized")

    @detailed_log_function(LogCategory.VECTOR_STORE)
    def _initialize_indexes(self):
        """Initialize FAISS indexes with optimal configuration"""
        index_logger.info(
            f"Initializing FAISS indexes with type: {self.index_type.value}"
        )

        try:
            if self.index_type == IndexType.FLAT:
                self.document_index = faiss.IndexFlatL2(self.dimension)
                self.entity_index = faiss.IndexFlatL2(self.dimension)

            elif self.index_type == IndexType.IVF:
                quantizer = faiss.IndexFlatL2(self.dimension)

            elif self.index_type == IndexType.IVFPQ:
                quantizer = faiss.IndexFlatL2(self.dimension)

            # Enable GPU if requested and available
            if self.enable_gpu and faiss.get_num_gpus() > 0:
                index_logger.info("Enabling GPU acceleration for FAISS indexes")
                gpu_resource = StandardGpuResources()

                self.document_index = index_cpu_to_gpu(
                    gpu_resource, 0, self.document_index
                )
                self.entity_index = index_cpu_to_gpu(gpu_resource, 0, self.entity_index)

                index_logger.info("GPU acceleration enabled")

            index_logger.info(
                "FAISS indexes initialized successfully",
                parameters={
                    "index_type": self.index_type.value,
                    "dimension": self.dimension,
                    "gpu_enabled": self.enable_gpu and faiss.get_num_gpus() > 0,
                },
            )

        except Exception as e:
            index_logger.error("Failed to initialize FAISS indexes", exception=e)
            raise

    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def _load_existing_data(self) -> None:
        """Load existing vectors and metadata from storage"""
        vector_logger.trace("Loading existing data from storage")

        # Load document index
        doc_index_path = self.document_index_path
        if doc_index_path.exists():
            try:
                self.document_index = faiss.read_index(str(doc_index_path))
                if self.enable_gpu and faiss.get_num_gpus() > 0:
                    gpu_res = StandardGpuResources()
                    self.document_index = index_cpu_to_gpu(gpu_res, 0, self.document_index)
                vector_logger.info(
                    "Document index loaded from disk",
                    parameters={"vectors_count": self.document_index.ntotal},
                )
            except Exception as e:
                vector_logger.warning("Failed to load document index", exception=e)

        # Load entity index
        entity_index_path = self.entity_index_path
        if entity_index_path.exists():
            try:
                self.entity_index = faiss.read_index(str(entity_index_path))
                if self.enable_gpu and faiss.get_num_gpus() > 0:
                    gpu_res = StandardGpuResources()
                    self.entity_index = index_cpu_to_gpu(gpu_res, 0, self.entity_index)
                vector_logger.info(
                    "Entity index loaded from disk",
                    parameters={"vectors_count": self.entity_index.ntotal},
                )
            except Exception as e:
                vector_logger.warning("Failed to load entity index", exception=e)

        # Load metadata cache
        await self._load_metadata_cache()

        # Update statistics
        doc_total = self.document_index.ntotal if self.document_index else 0
        ent_total = self.entity_index.ntotal if self.entity_index else 0
        self.statistics.total_vectors = doc_total + ent_total

        # Build index mapping from loaded metadata cache
        self.index_id_map.clear()
        self.vector_id_to_index.clear()
        for idx, vector_id in enumerate(self.metadata_cache.keys()):
            self.index_id_map[idx] = vector_id
            self.vector_id_to_index[vector_id] = idx

        vector_logger.info(
            "Existing data loaded successfully",
            parameters={
                "total_vectors": self.statistics.total_vectors,
                "document_vectors": doc_total,
                "entity_vectors": ent_total,
            },
        )

    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def _load_metadata_cache(self) -> None:
        """Load frequently accessed metadata into memory cache."""
        cache_logger.trace("Loading metadata cache")

        if not self.pool:
            raise RuntimeError("ConnectionPool is required for metadata loading")

        try:
            async with self.pool.get_pg_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM vector_metadata
                    ORDER BY last_accessed DESC
                    LIMIT 1000
                    """
                )

                for row in rows:
                    metadata = VectorMetadata(
                        vector_id=row["vector_id"],
                        document_id=row["document_id"],
                        content_hash=row["content_hash"],
                        content_preview=row["content_preview"],
                        vector_norm=row["vector_norm"],
                        dimension=row["dimension"],
                        created_at=row["created_at"],
                        last_accessed=row["last_accessed"],
                        access_count=row["access_count"],
                        source_file=row["source_file"],
                        document_type=row["document_type"],
                        tags=row["tags"] or [],
                        confidence_score=row["confidence_score"],
                        embedding_model=row["embedding_model"],
                        custom_metadata=row["custom_metadata"] or {},
                    )

                    self.metadata_cache[row["vector_id"]] = metadata

            cache_logger.info(
                "Metadata cache loaded",
                parameters={"cached_items": len(self.metadata_cache)},
            )

        except Exception as e:
            cache_logger.error("Failed to load metadata cache", exception=e)

    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def add_document(
        self,
        document_id: str,
        content: str,
        source_file: Optional[str] = None,
        document_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add document to vector store with comprehensive logging"""
        vector_logger.info(f"Adding document to vector store: {document_id}")

        if tags is None:
            tags = []
        if custom_metadata is None:
            custom_metadata = {}

        start_time = time.time()

        try:
            async with self._async_lock:
                # Generate content hash for deduplication
                content_hash = hashlib.sha256(content.encode()).hexdigest()

                # Check for duplicate content
                existing_vector = await self._find_by_content_hash(content_hash)
                if existing_vector:
                    vector_logger.warning(
                        f"Duplicate content detected for document {document_id}",
                        parameters={"existing_vector_id": existing_vector},
                    )
                    return existing_vector

                # Generate embedding
                vector_logger.trace("Generating embedding for document content")
                embedding = self.embedding_provider.embed_text(content)
                vector_norm = float(np.linalg.norm(embedding))

                # Generate unique vector ID
                vector_id = f"doc_{document_id}_{int(time.time())}"

                # Create metadata
                metadata = VectorMetadata(
                    vector_id=vector_id,
                    document_id=document_id,
                    content_hash=content_hash,
                    content_preview=content[:200],
                    vector_norm=vector_norm,
                    dimension=len(embedding),
                    source_file=source_file,
                    document_type=document_type,
                    tags=tags,
                    embedding_model=self.embedding_provider.model_name,
                    custom_metadata=custom_metadata,
                )

                # Add to FAISS index
                self.document_index.add(embedding.reshape(1, -1))
                new_idx = self.document_index.ntotal - 1
                self.index_id_map[new_idx] = vector_id
                self.vector_id_to_index[vector_id] = new_idx

                # Store metadata
                await self._store_metadata(metadata)

                # Update cache
                self.metadata_cache[vector_id] = metadata

                # Update statistics
                self.statistics.total_vectors += 1

                processing_time = time.time() - start_time

                return vector_id

        except Exception as e:
            vector_logger.error(f"Failed to add document {document_id}", exception=e)
            raise

    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def search_similar(
        self,
        query: str,
        k: int = 10,
        search_type: str = "document",
        min_similarity: float = 0.0,
        filters: Dict[str, Any] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors with comprehensive logging and filtering"""
        search_logger.info(f"Starting similarity search: '{query[:50]}...'")

        if filters is None:
            filters = {}

        search_start_time = time.time()

        try:
            async with self._async_lock:
                self.state = VectorStoreState.SEARCHING

                # Generate query embedding
                search_logger.trace("Generating query embedding")
                query_embedding = self.embedding_provider.embed_text(query)

                # Select appropriate index
                if search_type == "document":
                    index = self.document_index
                    index_name = "document_index"
                elif search_type == "entity":
                    index = self.entity_index
                    index_name = "entity_index"
                else:
                    raise ValueError(f"Invalid search type: {search_type}")

                if index is None:
                    raise RuntimeError(f"{index_name} is not initialized")

                search_logger.trace(
                    f"Using {index_name} for search",
                    parameters={"total_vectors": index.ntotal, "k": k},
                )

                results = []
                for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
                    if idx == -1:
                        break


                    vector_id = self._get_vector_id_by_index(idx, search_type)
                    if not vector_id:
                        continue


                search_time = time.time() - search_start_time
                self.state = VectorStoreState.READY

                # Update statistics
                self.statistics.total_searches += 1
                if self.statistics.average_search_time == 0:
                    self.statistics.average_search_time = search_time
                else:
                    self.statistics.average_search_time = (
                        self.statistics.average_search_time + search_time
                    ) / 2

                # Log search history
                search_record = {
                    "timestamp": datetime.now().isoformat(),
                    "query": query[:100],  # Truncate for privacy
                    "search_type": search_type,
                    "k": k,
                    "results_found": len(results),
                    "search_time": search_time,
                    "filters_applied": bool(filters),
                }

                self.search_history.append(search_record)
                if len(self.search_history) > 1000:
                    self.search_history = self.search_history[
                        -500:
                    ]  # Keep recent history

                search_logger.info(
                    "Search completed successfully",
                    parameters={
                        "query_length": len(query),
                        "results_found": len(results),
                        "search_time": search_time,
                        "index_used": index_name,
                        "filters_applied": len(filters),
                    },
                )

                performance_logger.performance_metric(
                    "Vector Search",
                    search_time,
                    {
                        "k": k,
                        "results_found": len(results),
                        "index_vectors": index.ntotal,
                    },
                )

                return results

        except Exception as e:
            self.state = VectorStoreState.READY
            search_logger.error(
                "Search failed",
                exception=e,
                parameters={
                    "query_length": len(query),
                    "search_type": search_type,
                    "k": k,
                },
            )
            raise

    # ------------------------------------------------------------------
    # Helper and maintenance methods

    def _start_background_optimization(self) -> None:
        """Start a simple background thread for optimization tasks."""
        thread = threading.Thread(
            target=self._optimization_worker, name="vector_opt", daemon=True
        )
        thread.start()

    def _optimization_worker(self) -> None:
        """Process queued optimization requests."""
        while True:
            try:
                task = self.optimization_queue.get(timeout=5)
            except Exception:
                continue
            if task is None:
                break
            # Placeholder for future optimization logic
            time.sleep(0.1)
            self.optimization_queue.task_done()


                """
                INSERT INTO vector_metadata (
                    vector_id, document_id, content_hash, content_preview,
                    vector_norm, dimension, created_at, last_accessed,
                    access_count, source_file, document_type, tags,
                    confidence_score, embedding_model, custom_metadata
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
                )
                ON CONFLICT (vector_id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    content_hash = EXCLUDED.content_hash,
                    content_preview = EXCLUDED.content_preview,
                    vector_norm = EXCLUDED.vector_norm,
                    dimension = EXCLUDED.dimension,
                    created_at = EXCLUDED.created_at,
                    last_accessed = EXCLUDED.last_accessed,
                    access_count = EXCLUDED.access_count,
                    source_file = EXCLUDED.source_file,
                    document_type = EXCLUDED.document_type,
                    tags = EXCLUDED.tags,
                    confidence_score = EXCLUDED.confidence_score,
                    embedding_model = EXCLUDED.embedding_model,
                    custom_metadata = EXCLUDED.custom_metadata
                """,
                metadata.vector_id,
                metadata.document_id,
                metadata.content_hash,
                metadata.content_preview,
                metadata.vector_norm,
                metadata.dimension,
                metadata.created_at,
                metadata.last_accessed,
                metadata.access_count,
                metadata.source_file,
                metadata.document_type,
                json.dumps(metadata.tags),
                metadata.confidence_score,
                metadata.embedding_model,
                json.dumps(metadata.custom_metadata),
            )

    async def _find_by_content_hash(self, content_hash: str) -> Optional[str]:
        """Return vector_id matching the given content hash if it exists."""
        for meta in self.metadata_cache.values():
            if meta.content_hash == content_hash:
                return meta.vector_id

        if not self.pool:
            raise RuntimeError("ConnectionPool is required for DB access")

        async with self.pool.get_pg_connection() as conn:
            row = await conn.fetchrow(
                "SELECT vector_id FROM vector_metadata WHERE content_hash=$1",
                content_hash,
            )
            return row["vector_id"] if row else None

    def _get_vector_id_by_index(self, idx: int, search_type: str) -> Optional[str]:
        """Lookup vector_id by FAISS index position."""
        return self.index_id_map.get(idx)

    async def _get_metadata(self, vector_id: str) -> Optional[VectorMetadata]:
        """Retrieve metadata from cache or database."""
        metadata = self.metadata_cache.get(vector_id)
        if metadata:
            return metadata

        if not self.pool:
            raise RuntimeError("ConnectionPool is required for DB access")

        async with self.pool.get_pg_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM vector_metadata WHERE vector_id=$1", vector_id
            )
            if not row:
                return None
            metadata = VectorMetadata(
                vector_id=row["vector_id"],
                document_id=row["document_id"],
                content_hash=row["content_hash"],
                content_preview=row["content_preview"],
                vector_norm=row["vector_norm"],
                dimension=row["dimension"],
                created_at=row["created_at"],
                last_accessed=row["last_accessed"],
                access_count=row["access_count"],
                source_file=row["source_file"],
                document_type=row["document_type"],
                tags=row["tags"] or [],
                confidence_score=row["confidence_score"],
                embedding_model=row["embedding_model"],
                custom_metadata=row["custom_metadata"] or {},
            )
            self.metadata_cache[vector_id] = metadata
            return metadata

    def _apply_filters(self, metadata: VectorMetadata, filters: Dict[str, Any]) -> bool:
        """Simple filtering implementation."""
        for key, value in filters.items():
            attr_val = getattr(metadata, key, None)
            if attr_val != value and metadata.custom_metadata.get(key) != value:
                return False
        return True

    async def _update_access_stats(self, vector_id: str) -> None:
        """Update access statistics for a vector."""
        metadata = self.metadata_cache.get(vector_id)
        if not metadata or not self.transaction_manager:
            return

        metadata.access_count += 1
        metadata.last_accessed = datetime.now(timezone.utc)

        async with self.transaction_manager.transaction() as conn:
            await conn.execute(
                "UPDATE vector_metadata SET last_accessed=$1, access_count=$2 WHERE vector_id=$3",
                metadata.last_accessed,
                metadata.access_count,
                vector_id,
            )

    def get_system_status(self) -> Dict[str, Any]:
        """Return a basic health summary of the vector store."""
        doc_total = self.document_index.ntotal if self.document_index else 0
        ent_total = self.entity_index.ntotal if self.entity_index else 0
        return {
            "state": self.state.value,
            "dimension": self.dimension,
            "index_type": self.index_type.value,
            "total_vectors": doc_total + ent_total,
            "cache_size": len(self.metadata_cache),
        }


def create_enhanced_vector_store(
    connection_pool: ConnectionPool,
    config: Optional[Dict[str, Any]] | None = None,
) -> "EnhancedVectorStore":
    """Factory function used by :class:`ServiceContainer`."""

    cfg = config or {}
    return EnhancedVectorStore(
        storage_path=cfg.get("STORAGE_PATH", "./storage/vectors"),
        embedding_model=cfg.get(
            "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        index_type=IndexType(cfg.get("DEFAULT_INDEX_TYPE", "HNSW")),
        enable_gpu=cfg.get("ENABLE_GPU_FAISS", False),
        document_index_path=cfg.get("DOCUMENT_INDEX_PATH"),
        entity_index_path=cfg.get("ENTITY_INDEX_PATH"),

    )

    # ------------------------------------------------------------------


if __name__ == "__main__":
