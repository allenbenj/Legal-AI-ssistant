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
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue

# Import detailed logging system
from .detailed_logging import (
    get_detailed_logger,
    LogCategory,
    detailed_log_function,
)

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
    FLAT = "IndexFlatL2"          # Exact search, high accuracy
    IVF = "IndexIVFFlat"          # Approximate search, good speed/accuracy
    HNSW = "IndexHNSWFlat"        # Hierarchical search, excellent speed
    PQ = "IndexPQ"                # Product quantization, memory efficient
    IVFPQ = "IndexIVFPQ"          # Combined IVF+PQ, balanced

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
        
        embedding_logger.info("Embedding provider initialized", parameters={
            'model_name': self.model_name,
            'dimension': self.dimension,
            'cache_enabled': True
        })
    
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
            
            embedding_logger.info("SentenceTransformers model loaded successfully", parameters={
                'model': self.model_name,
                'dimension': self.dimension
            })
            return
            
        except ImportError:
            embedding_logger.warning("SentenceTransformers not available, trying Ollama")
            
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
            
            embedding_logger.trace("Embedding retrieved from cache", parameters={
                'text_hash': text_hash[:8],
                'cache_time': cache_time,
                'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses)
            })
            
            return embedding
        
        self.cache_misses += 1
        
        try:
            if isinstance(self.model, object) and hasattr(self.model, 'encode'):
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
                cache_logger.info("Embedding cache pruned", parameters={'new_size': len(self.embedding_cache)})
            
            embedding_time = time.time() - start_time
            self.total_embeddings += 1
            self.total_embedding_time += embedding_time
            
            embedding_logger.info(
                "Embedding generated successfully",
                parameters={
                    'text_length': len(text),
                    'embedding_dimension': len(embedding),
                    'embedding_time': embedding_time,
                    'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses),
                    'total_embeddings': self.total_embeddings,
                },
            )
            
            return embedding
            
        except Exception as e:
            embedding_logger.error("Embedding generation failed", exception=e, parameters={
                'text_length': len(text),
                'model_type': str(type(self.model))
            })
            raise
    
    @detailed_log_function(LogCategory.VECTOR_STORE)
    def _get_ollama_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama API"""
        import requests
        
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=30
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
        
        for i, char in enumerate(text[:self.dimension]):
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
        enable_gpu: bool = False
    ):
        """Initialize enhanced vector store with comprehensive configuration"""
        vector_logger.info("=== INITIALIZING ENHANCED VECTOR STORE ===")
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.state = VectorStoreState.INITIALIZING
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
        self._search_lock = threading.Lock()
        
        # Initialize storage
        self._initialize_storage()
        self._initialize_indexes()
        self._load_existing_data()
        
        # Start background optimization
        self._start_background_optimization()
        
        self.state = VectorStoreState.READY
        
        vector_logger.info("Enhanced vector store initialization complete", parameters={
            'storage_path': str(self.storage_path),
            'dimension': self.dimension,
            'index_type': self.index_type.value,
            'gpu_enabled': self.enable_gpu,
            'total_vectors': self.statistics.total_vectors
        })
    
    @detailed_log_function(LogCategory.VECTOR_STORE)
    def _initialize_storage(self):
        """Initialize SQLite storage for metadata with detailed schema"""
        vector_logger.trace("Initializing metadata storage")
        
        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_metadata (
                    vector_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    content_preview TEXT,
                    vector_norm REAL,
                    dimension INTEGER,
                    created_at TEXT,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 0,
                    source_file TEXT,
                    document_type TEXT,
                    tags TEXT,
                    confidence_score REAL DEFAULT 1.0,
                    embedding_model TEXT,
                    custom_metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_id ON vector_metadata(document_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash ON vector_metadata(content_hash)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON vector_metadata(last_accessed)
            """)
            
            conn.commit()
        
        vector_logger.info("Metadata storage initialized")
    
    @detailed_log_function(LogCategory.VECTOR_STORE)
    def _initialize_indexes(self):
        """Initialize FAISS indexes with optimal configuration"""
        index_logger.info(f"Initializing FAISS indexes with type: {self.index_type.value}")
        
        try:
            if self.index_type == IndexType.FLAT:
                # Exact search index
                self.document_index = faiss.IndexFlatL2(self.dimension)
                self.entity_index = faiss.IndexFlatL2(self.dimension)
                
            elif self.index_type == IndexType.IVF:
                # Inverted file index for approximate search
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.document_index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
                self.entity_index = faiss.IndexIVFFlat(quantizer, self.dimension, 50)
                
            elif self.index_type == IndexType.HNSW:
                # Hierarchical Navigable Small World for fast search
                self.document_index = faiss.IndexHNSWFlat(self.dimension, 32)
                self.entity_index = faiss.IndexHNSWFlat(self.dimension, 32)
                
            elif self.index_type == IndexType.PQ:
                # Product quantization for memory efficiency
                self.document_index = faiss.IndexPQ(self.dimension, 8, 8)
                self.entity_index = faiss.IndexPQ(self.dimension, 8, 8)
                
            elif self.index_type == IndexType.IVFPQ:
                # Combined IVF and PQ
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.document_index = faiss.IndexIVFPQ(quantizer, self.dimension, 100, 8, 8)
                self.entity_index = faiss.IndexIVFPQ(quantizer, self.dimension, 50, 8, 8)
            
            # Enable GPU if requested and available
            if self.enable_gpu and faiss.get_num_gpus() > 0:
                index_logger.info("Enabling GPU acceleration for FAISS indexes")
                gpu_resource = StandardGpuResources()

                self.document_index = index_cpu_to_gpu(gpu_resource, 0, self.document_index)
                self.entity_index = index_cpu_to_gpu(gpu_resource, 0, self.entity_index)
                
                index_logger.info("GPU acceleration enabled")
            
            index_logger.info("FAISS indexes initialized successfully", parameters={
                'index_type': self.index_type.value,
                'dimension': self.dimension,
                'gpu_enabled': self.enable_gpu and faiss.get_num_gpus() > 0
            })
            
        except Exception as e:
            index_logger.error("Failed to initialize FAISS indexes", exception=e)
            raise
    
    @detailed_log_function(LogCategory.VECTOR_STORE)
    def _load_existing_data(self):
        """Load existing vectors and metadata from storage"""
        vector_logger.trace("Loading existing data from storage")
        
        # Load document index
        doc_index_path = self.storage_path / "document_index.faiss"
        if doc_index_path.exists():
            try:
                self.document_index = faiss.read_index(str(doc_index_path))
                vector_logger.info(
                    "Document index loaded from disk",
                    parameters={"vectors_count": self.document_index.ntotal},
                )
            except Exception as e:
                vector_logger.warning("Failed to load document index", exception=e)
        
        # Load entity index
        entity_index_path = self.storage_path / "entity_index.faiss"
        if entity_index_path.exists():
            try:
                self.entity_index = faiss.read_index(str(entity_index_path))
                vector_logger.info(
                    "Entity index loaded from disk",
                    parameters={"vectors_count": self.entity_index.ntotal},
                )
            except Exception as e:
                vector_logger.warning("Failed to load entity index", exception=e)
        
        # Load metadata cache
        self._load_metadata_cache()
        
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
        
        vector_logger.info("Existing data loaded successfully", parameters={
            'total_vectors': self.statistics.total_vectors,
            'document_vectors': doc_total,
            'entity_vectors': ent_total
        })
    
    @detailed_log_function(LogCategory.VECTOR_STORE)
    def _load_metadata_cache(self):
        """Load frequently accessed metadata into memory cache"""
        cache_logger.trace("Loading metadata cache")
        
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                # Load most recently accessed metadata
                cursor = conn.execute("""
                    SELECT * FROM vector_metadata 
                    ORDER BY last_accessed DESC 
                    LIMIT 1000
                """)
                
                for row in cursor:
                    metadata = VectorMetadata(
                        vector_id=row[0],
                        document_id=row[1],
                        content_hash=row[2],
                        content_preview=row[3],
                        vector_norm=row[4],
                        dimension=row[5],
                        created_at=datetime.fromisoformat(row[6]),
                        last_accessed=datetime.fromisoformat(row[7]),
                        access_count=row[8],
                        source_file=row[9],
                        document_type=row[10],
                        tags=json.loads(row[11]) if row[11] else [],
                        confidence_score=row[12],
                        embedding_model=row[13],
                        custom_metadata=json.loads(row[14]) if row[14] else {}
                    )
                    
                    self.metadata_cache[row[0]] = metadata
            
            cache_logger.info("Metadata cache loaded", parameters={'cached_items': len(self.metadata_cache)})
            
        except Exception as e:
            cache_logger.error("Failed to load metadata cache", exception=e)
    
    @detailed_log_function(LogCategory.VECTOR_STORE)
    def add_document(
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
            with self._lock:
                # Generate content hash for deduplication
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                
                # Check for duplicate content
                existing_vector = self._find_by_content_hash(content_hash)
                if existing_vector:
                    vector_logger.warning(f"Duplicate content detected for document {document_id}",
                                        parameters={'existing_vector_id': existing_vector})
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
                    custom_metadata=custom_metadata
                )
                
                # Add to FAISS index
                self.document_index.add(embedding.reshape(1, -1))
                new_idx = self.document_index.ntotal - 1
                self.index_id_map[new_idx] = vector_id
                self.vector_id_to_index[vector_id] = new_idx
                
                # Store metadata
                self._store_metadata(metadata)
                
                # Update cache
                self.metadata_cache[vector_id] = metadata
                
                # Update statistics
                self.statistics.total_vectors += 1
                
                processing_time = time.time() - start_time
                
                vector_logger.info("Document added successfully", parameters={
                    'vector_id': vector_id,
                    'document_id': document_id,
                    'content_length': len(content),
                    'vector_norm': vector_norm,
                    'processing_time': processing_time,
                    'total_vectors': self.statistics.total_vectors
                })
                
                performance_logger.performance_metric("Document Addition", processing_time, {
                    'content_length': len(content),
                    'vector_dimension': len(embedding)
                })
                
                return vector_id
                
        except Exception as e:
            vector_logger.error(f"Failed to add document {document_id}", exception=e)
            raise
    
    @detailed_log_function(LogCategory.VECTOR_STORE)
    def search_similar(
        self,
        query: str,
        k: int = 10,
        search_type: str = "document",
        min_similarity: float = 0.0,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Search for similar vectors with comprehensive logging and filtering"""
        search_logger.info(f"Starting similarity search: '{query[:50]}...'")
        
        if filters is None:
            filters = {}
        
        search_start_time = time.time()
        
        try:
            with self._search_lock:
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
                    parameters={'total_vectors': index.ntotal, 'k': k},
                )
                
                # Perform search
                distances, indices = index.search(query_embedding.reshape(1, -1), k * 2)  # Get extra for filtering
                
                # Process results
                results = []
                for _, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx == -1:  # No more results
                        break
                    
                    # Calculate similarity score (convert distance to similarity)
                    similarity_score = 1.0 / (1.0 + distance)
                    
                    if similarity_score < min_similarity:
                        continue
                    
                    # Get metadata for this vector
                    vector_id = self._get_vector_id_by_index(idx, search_type)
                    if not vector_id:
                        continue
                    
                    metadata = self._get_metadata(vector_id)
                    if not metadata:
                        continue
                    
                    # Apply filters
                    if not self._apply_filters(metadata, filters):
                        continue
                    
                    # Update access statistics
                    self._update_access_stats(vector_id)
                    
                    # Create search result
                    result = SearchResult(
                        vector_id=vector_id,
                        document_id=metadata.document_id,
                        content_preview=metadata.content_preview,
                        similarity_score=similarity_score,
                        distance=float(distance),
                        metadata=metadata,
                        search_time=time.time() - search_start_time,
                        index_used=index_name,
                        rank=len(results) + 1
                    )
                    
                    results.append(result)
                    
                    if len(results) >= k:
                        break
                
                search_time = time.time() - search_start_time
                self.state = VectorStoreState.READY
                
                # Update statistics
                self.statistics.total_searches += 1
                if self.statistics.average_search_time == 0:
                    self.statistics.average_search_time = search_time
                else:
                    self.statistics.average_search_time = (self.statistics.average_search_time + search_time) / 2
                
                # Log search history
                search_record = {
                    'timestamp': datetime.now().isoformat(),
                    'query': query[:100],  # Truncate for privacy
                    'search_type': search_type,
                    'k': k,
                    'results_found': len(results),
                    'search_time': search_time,
                    'filters_applied': bool(filters)
                }
                
                self.search_history.append(search_record)
                if len(self.search_history) > 1000:
                    self.search_history = self.search_history[-500:]  # Keep recent history
                
                search_logger.info("Search completed successfully", parameters={
                    'query_length': len(query),
                    'results_found': len(results),
                    'search_time': search_time,
                    'index_used': index_name,
                    'filters_applied': len(filters)
                })
                
                performance_logger.performance_metric("Vector Search", search_time, {
                    'k': k,
                    'results_found': len(results),
                    'index_vectors': index.ntotal
                })
                
                return results
                
        except Exception as e:
            self.state = VectorStoreState.READY
            search_logger.error("Search failed", exception=e, parameters={
                'query_length': len(query),
                'search_type': search_type,
                'k': k
            })
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

    def _store_metadata(self, metadata: VectorMetadata) -> None:
        """Persist metadata to the SQLite database."""
        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO vector_metadata (
                    vector_id, document_id, content_hash, content_preview,
                    vector_norm, dimension, created_at, last_accessed,
                    access_count, source_file, document_type, tags,
                    confidence_score, embedding_model, custom_metadata
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    metadata.vector_id,
                    metadata.document_id,
                    metadata.content_hash,
                    metadata.content_preview,
                    metadata.vector_norm,
                    metadata.dimension,
                    metadata.created_at.isoformat(),
                    metadata.last_accessed.isoformat(),
                    metadata.access_count,
                    metadata.source_file,
                    metadata.document_type,
                    json.dumps(metadata.tags),
                    metadata.confidence_score,
                    metadata.embedding_model,
                    json.dumps(metadata.custom_metadata),
                ),
            )
            conn.commit()

    def _find_by_content_hash(self, content_hash: str) -> Optional[str]:
        """Return vector_id matching the given content hash if it exists."""
        for meta in self.metadata_cache.values():
            if meta.content_hash == content_hash:
                return meta.vector_id
        with sqlite3.connect(self.metadata_db_path) as conn:
            cursor = conn.execute(
                "SELECT vector_id FROM vector_metadata WHERE content_hash=?",
                (content_hash,),
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def _get_vector_id_by_index(self, idx: int, search_type: str) -> Optional[str]:
        """Lookup vector_id by FAISS index position."""
        return self.index_id_map.get(idx)

    def _get_metadata(self, vector_id: str) -> Optional[VectorMetadata]:
        """Retrieve metadata from cache or database."""
        metadata = self.metadata_cache.get(vector_id)
        if metadata:
            return metadata
        with sqlite3.connect(self.metadata_db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM vector_metadata WHERE vector_id=?", (vector_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            metadata = VectorMetadata(
                vector_id=row[0],
                document_id=row[1],
                content_hash=row[2],
                content_preview=row[3],
                vector_norm=row[4],
                dimension=row[5],
                created_at=datetime.fromisoformat(row[6]),
                last_accessed=datetime.fromisoformat(row[7]),
                access_count=row[8],
                source_file=row[9],
                document_type=row[10],
                tags=json.loads(row[11]) if row[11] else [],
                confidence_score=row[12],
                embedding_model=row[13],
                custom_metadata=json.loads(row[14]) if row[14] else {},
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

    def _update_access_stats(self, vector_id: str) -> None:
        """Update access statistics for a vector."""
        metadata = self.metadata_cache.get(vector_id)
        if not metadata:
            return
        metadata.access_count += 1
        metadata.last_accessed = datetime.now()
        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.execute(
                "UPDATE vector_metadata SET last_accessed=?, access_count=? WHERE vector_id=?",
                (
                    metadata.last_accessed.isoformat(),
                    metadata.access_count,
                    vector_id,
                ),
            )
            conn.commit()

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
    service_container: Any,
    service_config: Optional[Dict[str, Any]] | None = None,
) -> "EnhancedVectorStore":
    """Factory function used by :class:`ServiceContainer`."""

    cfg = service_config or {}
    return EnhancedVectorStore(
        storage_path=cfg.get("STORAGE_PATH", "./storage/vectors"),
        embedding_model=cfg.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        index_type=IndexType(cfg.get("DEFAULT_INDEX_TYPE", "HNSW")),
        enable_gpu=cfg.get("ENABLE_GPU_FAISS", False),
    )

    # ------------------------------------------------------------------

if __name__ == "__main__":
    # Test the enhanced vector store
    vector_logger.info("Testing enhanced vector store")
    
    store = EnhancedVectorStore()
    
    # Test document addition
    doc_id = store.add_document(
        "test_doc_1",
        "This is a test legal document about contract violations and legal proceedings.",
        source_file="test.pdf",
        document_type="legal_brief",
        tags=["contract", "violation", "legal"]
    )
    
    # Test search
    results = store.search_similar("contract violations", k=5)
    
    print(f"Added document: {doc_id}")
    print(f"Search results: {len(results)}")
    
    # Get system status
    status = store.get_system_status()
    print(f"Vector store status: {status}")
