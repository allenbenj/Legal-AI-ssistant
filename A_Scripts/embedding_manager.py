# legal_ai_system/core/embedding_manager.py

# Embedding Manager - Centralized Embedding Operations
# This module provides the EmbeddingManager class for handling all embedding
# operations across the legal AI system.

import asyncio
import hashlib
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path
import threading
import json  # For logging complex objects
import math
from datetime import datetime

# Use detailed_logging
from .detailed_logging import get_detailed_logger, LogCategory, detailed_log_function
# Import the ABC and a concrete implementation from the knowledge.vector_store package
from ..knowledge.vector_store.vector_store import EmbeddingProviderVS, SentenceTransformerEmbeddingProvider
from ..core.unified_exceptions import ConfigurationError  # For init errors
from .embeddings.providers import EmbeddingProviderVS, SentenceTransformerEmbeddingProvider # Corrected import

# Initialize logger for this module
embedding_manager_logger = get_detailed_logger("EmbeddingManager", LogCategory.VECTOR_STORE)

# Numpy is optional for similarity calculations
try:
    import numpy as np_  # Alias to avoid conflicts
    NUMPY_AVAILABLE = True
    embedding_manager_logger.debug("NumPy library found and will be used for vector operations.")
except ImportError:
    embedding_manager_logger.warning("NumPy library not found. Using pure Python for vector operations (less efficient).")
    NUMPY_AVAILABLE = False

# Removed EmbeddingClient import and fallback as it's not consistently used
class EmbeddingManager:
    """
    Centralized manager for embedding operations.
    Handles caching, batching, and provider management for embeddings.
    """
    
    @detailed_log_function(LogCategory.VECTOR_STORE)
    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProviderVS] = None,  # Allow injecting a provider
        default_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",  # Used if provider is None
        cache_enabled: bool = True,
        cache_dir_str: str = "./storage/embeddings_cache_manager",  # Specific cache for manager
        batch_size: int = 32,
        service_config: Optional[Dict[str, Any]] = None  # For future config loading
    ):
        config = service_config or {}
        embedding_manager_logger.info("Initializing EmbeddingManager.", 
                                      parameters={'default_model': default_model_name, 'cache': cache_enabled})
        
        self._embedding_provider_instance = embedding_provider  # Store injected provider
        self.default_model_name = default_model_name  # Used if creating own provider
        
        self.cache_enabled = bool(config.get("cache_enabled", cache_enabled))
        self.cache_dir = Path(config.get("cache_dir_str", cache_dir_str))
        self.batch_size = int(config.get("batch_size", batch_size))
        
        self._cache: Dict[str, List[float]] = {}
        self._cache_lock = threading.RLock()
        self._initialized = False
        self._active_provider_model_name: Optional[str] = None

    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def initialize(self) -> None:
        if self._initialized:
            embedding_manager_logger.warning("EmbeddingManager already initialized.")
            return

        embedding_manager_logger.info("Starting EmbeddingManager initialization.")
        try:
            if self._embedding_provider_instance is None:
                embedding_manager_logger.info(f"No embedding provider injected, creating default SentenceTransformerEmbeddingProvider.")
                self._embedding_provider_instance = SentenceTransformerEmbeddingProvider(model_name=self.default_model_name)
            
            if not hasattr(self._embedding_provider_instance, 'initialize') or \
               not hasattr(self._embedding_provider_instance, 'embed_texts') or \
               not hasattr(self._embedding_provider_instance, 'dimension'):
                msg = "Provided embedding_provider does not conform to EmbeddingProviderVS interface."
                embedding_manager_logger.error(msg)
                raise ConfigurationError(msg, config_key="embedding_provider_instance")

            await self._embedding_provider_instance.initialize()  # Initialize the provider
            self._active_provider_model_name = self._embedding_provider_instance.model_name

            if self.cache_enabled:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                await self._load_cache_async()
            
            self._initialized = True
            embedding_manager_logger.info("EmbeddingManager initialized successfully.", 
                                          parameters={'active_provider_model': self._active_provider_model_name})
        except Exception as e:
            embedding_manager_logger.critical("Failed to initialize EmbeddingManager.", exception=e)
            self._initialized = False
            raise

    def _get_active_model_name_for_cache(self) -> str:
        """Returns the model name of the active provider for cache key generation."""
        if self._embedding_provider_instance and hasattr(self._embedding_provider_instance, 'model_name'):
            return self._embedding_provider_instance.model_name
        return self.default_model_name  # Fallback
        
    async def _load_cache_async(self) -> None:
        cache_model_name = self._get_active_model_name_for_cache()
        cache_file = self.cache_dir / f"{cache_model_name.replace('/', '_')}_embedding_cache.pkl"
        embedding_manager_logger.debug("Attempting to load embedding cache.", parameters={'cache_file': str(cache_file)})
        
        if cache_file.exists():
            try:
                loop = asyncio.get_event_loop()
                with self._cache_lock:
                    loaded_cache = await loop.run_in_executor(
                        None, self._load_cache_sync_op, cache_file
                    )
                    if loaded_cache is not None:
                        self._cache = loaded_cache or {}
                        embedding_manager_logger.info(f"Loaded {len(self._cache)} cached embeddings.", parameters={'cache_file': str(cache_file)})
                    else:
                        self._cache = {}
            except Exception as e:
                embedding_manager_logger.warning(f"Failed to load embedding cache.", parameters={'cache_file': str(cache_file)}, exception=e)
                self._cache = {}
        else:
            embedding_manager_logger.info("No existing embedding cache file found.", parameters={'cache_file': str(cache_file)})

    def _load_cache_sync_op(self, cache_file: Path) -> Optional[Dict[str, List[float]]]:
        """Synchronous part of cache loading for executor."""
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    return data
                else:
                    embedding_manager_logger.warning("Cache file format error: expected a dict.", parameters={'cache_file': str(cache_file)})
                    return {}
        except (pickle.UnpicklingError, EOFError, FileNotFoundError, TypeError) as e:
            embedding_manager_logger.warning(f"Error unpickling cache file, creating new cache.", parameters={'cache_file': str(cache_file)}, exception=e)
            return {}

    async def _save_cache_async(self) -> None:
        """Save embedding cache to disk asynchronously."""
        if not self.cache_enabled:
            return
        
        cache_model_name = self._get_active_model_name_for_cache()
        cache_file = self.cache_dir / f"{cache_model_name.replace('/', '_')}_embedding_cache.pkl"
        embedding_manager_logger.debug("Attempting to save embedding cache.", parameters={'cache_file': str(cache_file), 'num_items': len(self._cache)})

        try:
            loop = asyncio.get_event_loop()
            with self._cache_lock:
                cache_copy = self._cache.copy()
            
            if not cache_copy:
                embedding_manager_logger.info("Embedding cache is empty, skipping save.", parameters={'cache_file': str(cache_file)})
                return

            await loop.run_in_executor(
                None, self._save_cache_sync_op, cache_file, cache_copy
            )
            embedding_manager_logger.info("Embedding cache saved successfully.", parameters={'cache_file': str(cache_file)})
        except Exception as e:
            embedding_manager_logger.warning(f"Failed to save embedding cache.", parameters={'cache_file': str(cache_file)}, exception=e)

    def _save_cache_sync_op(self, cache_file: Path, cache_data: Dict[str, List[float]]):
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
            
    def _get_cache_key(self, text: str) -> str:
        cache_model_name = self._get_active_model_name_for_cache()
        content = f"{cache_model_name}:{text}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts, using cache and batching."""
        if not self._initialized:
            await self.initialize()
            if not self._initialized:
                embedding_manager_logger.error("EmbeddingManager not initialized. Cannot embed texts.")
                raise RuntimeError("EmbeddingManager not initialized.")

        if not texts:
            return []
        
        embedding_manager_logger.debug(f"Request to embed {len(texts)} texts.", parameters={'batch_size': self.batch_size})

        results_ordered: List[Optional[List[float]]] = [None] * len(texts)
        uncached_texts_list: List[str] = []
        uncached_indices: List[int] = []

        if self.cache_enabled:
            with self._cache_lock:
                for i, text_item in enumerate(texts):
                    if not text_item or not text_item.strip():
                        results_ordered[i] = []
                        embedding_manager_logger.trace("Skipping embedding for empty text.", parameters={'index': i})
                        continue
                    cache_key = self._get_cache_key(text_item)
                    if cache_key in self._cache:
                        results_ordered[i] = self._cache[cache_key]
                        embedding_manager_logger.trace("Cache hit.", parameters={'text_preview': text_item[:30]+"..."})
                    else:
                        uncached_indices.append(i)
                        uncached_texts_list.append(text_item)
                        embedding_manager_logger.trace("Cache miss.", parameters={'text_preview': text_item[:30]+"..."})
        else:
            for i, text_item in enumerate(texts):
                if not text_item or not text_item.strip():
                    results_ordered[i] = []
                    continue
                uncached_indices.append(i)
                uncached_texts_list.append(text_item)

        if uncached_texts_list:
            embedding_manager_logger.info(f"Generating embeddings for {len(uncached_texts_list)} new texts.")
            loop = asyncio.get_event_loop()
            
            all_new_embeddings: List[List[float]] = []
            for i in range(0, len(uncached_texts_list), self.batch_size):
                batch_to_embed = uncached_texts_list[i:i + self.batch_size]
                embedding_manager_logger.debug(f"Processing batch of {len(batch_to_embed)} texts.")
                try:
                    batch_embeddings_result = await loop.run_in_executor(
                        None, self._embedding_provider_instance.embed_texts, batch_to_embed
                    )
                    batch_embeddings_list = [
                        emb.tolist() if hasattr(emb, 'tolist') else (emb if isinstance(emb, list) else [])
                        for emb in batch_embeddings_result
                    ]
                    all_new_embeddings.extend(batch_embeddings_list)
                except Exception as e:
                    embedding_manager_logger.error(f"Error embedding batch.",
                                                  parameters={'batch_preview': batch_to_embed[0][:50]+"...", 'error': str(e)},
                                                  exception=e)
                    all_new_embeddings.extend([[] for _ in batch_to_embed])

            with self._cache_lock:
                for original_idx, text_content, new_embedding in zip(uncached_indices, uncached_texts_list, all_new_embeddings):
                    results_ordered[original_idx] = new_embedding
                    if self.cache_enabled and new_embedding:
                        cache_key = self._get_cache_key(text_content)
                        self._cache[cache_key] = new_embedding
            
            if self.cache_enabled and any(all_new_embeddings):
                asyncio.create_task(self._save_cache_async())
        
        final_embeddings = [emb if emb is not None else [] for emb in results_ordered]
        embedding_manager_logger.info(f"Embedding generation complete for {len(texts)} texts.",
                                      parameters={'newly_embedded': len(uncached_texts_list), 'from_cache': len(texts) - len(uncached_texts_list)})
        return final_embeddings
            
    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def embed_documents(self, documents: List[Dict[str, Any]], text_field: str = 'content') -> List[Dict[str, Any]]:
        """Add 'embedding' field to a list of document dictionaries."""
        if not self._initialized:
            await self.initialize()
            if not self._initialized:
                embedding_manager_logger.error("EmbeddingManager not initialized. Cannot embed documents.")
                raise RuntimeError("EmbeddingManager not initialized.")

        embedding_manager_logger.info(f"Embedding {len(documents)} documents.", parameters={'text_field': text_field})
        
        texts_to_embed_map: Dict[int, str] = {}
        for i, doc in enumerate(documents):
            text = doc.get(text_field, "")
            if not isinstance(text, str):
                embedding_manager_logger.warning("Document text_field is not a string, attempting conversion.",
                                                parameters={'doc_id': doc.get('id', 'unknown'), 'field_type': type(text).__name__})
                text = str(text)
            if text.strip():
                texts_to_embed_map[i] = text
        
        texts_list = list(texts_to_embed_map.values())
        original_indices_list = list(texts_to_embed_map.keys())
        
        generated_embeddings = await self.embed_texts(texts_list) if texts_list else []
        
        enhanced_documents = [doc.copy() for doc in documents]
        
        for i, original_idx in enumerate(original_indices_list):
            if i < len(generated_embeddings):
                enhanced_documents[original_idx]['embedding'] = generated_embeddings[i]
            else:
                enhanced_documents[original_idx]['embedding'] = []
                embedding_manager_logger.error("Mismatch in generated embeddings count for documents.", parameters={'doc_original_index': original_idx})
        
        for i in range(len(enhanced_documents)):
            if 'embedding' not in enhanced_documents[i]:
                enhanced_documents[i]['embedding'] = []

        embedding_manager_logger.info(f"Embeddings added to {len(enhanced_documents)} documents.")
        return enhanced_documents
        
    async def _ensure_initialized(self):
        if not self._initialized or not self._embedding_provider_instance:
            await self.initialize()
            if not self._initialized or not self._embedding_provider_instance:
                msg = "EmbeddingManager or its provider is not initialized."
                embedding_manager_logger.critical(msg)
                raise RuntimeError(msg)
    
    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def embed_text(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        await self._ensure_initialized()
        embeddings = await self.embed_texts([text])
        return embeddings[0] if embeddings else []
        
    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def compute_similarity(
        self,
        text1: str,
        text2: str,
        similarity_type: str = "cosine"
    ) -> float:
        """Compute similarity between two texts."""
        if not self._initialized:
            await self.initialize()
            if not self._initialized:
                raise RuntimeError("EmbeddingManager not initialized.")

        embedding_manager_logger.debug("Computing similarity.", parameters={'type': similarity_type, 'text1_len': len(text1), 'text2_len': len(text2)})
        try:
            embeddings = await self.embed_texts([text1, text2])
            if len(embeddings) != 2 or not embeddings[0] or not embeddings[1]:
                embedding_manager_logger.warning("Could not generate embeddings for similarity.", parameters={'text1_empty': not embeddings[0], 'text2_empty': not embeddings[1]})
                return 0.0
            
            vec1, vec2 = embeddings[0], embeddings[1]

            if NUMPY_AVAILABLE:
                vec1_np, vec2_np = np_.array(vec1), np_.array(vec2)
                if similarity_type == "cosine":
                    dot = np_.dot(vec1_np, vec2_np)
                    norm1, norm2 = np_.linalg.norm(vec1_np), np_.linalg.norm(vec2_np)
                    return float(dot / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0
                elif similarity_type == "euclidean":
                    dist = np_.linalg.norm(vec1_np - vec2_np)
                    return float(1.0 / (1.0 + dist))
                else:
                    raise ValueError(f"Unsupported similarity type: {similarity_type}")
            else:
                if similarity_type == "cosine":
                    return self._cosine_similarity_py(vec1, vec2)
                elif similarity_type == "euclidean":
                    return self._euclidean_similarity_py(vec1, vec2)
                else:
                    raise ValueError(f"Unsupported similarity type: {similarity_type}")
        except Exception as e:
            embedding_manager_logger.error("Failed to compute similarity.", exception=e)
            return 0.0
    
    def _cosine_similarity_py(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        return dot_product / (magnitude1 * magnitude2) if magnitude1 > 0 and magnitude2 > 0 else 0.0

    def _euclidean_similarity_py(self, vec1: List[float], vec2: List[float]) -> float:
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
        return 1.0 / (1.0 + distance)

    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def find_similar_texts(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        if not self._initialized:
            await self.initialize()
            if not self._initialized:
                raise RuntimeError("EmbeddingManager not initialized.")

        embedding_manager_logger.debug(f"Finding {top_k} similar texts.",
                                      parameters={'num_candidates': len(candidate_texts), 'query_len': len(query_text)})
        try:
            if not query_text.strip() or not candidate_texts:
                return []

            query_embedding = await self.embed_text(query_text)
            if not query_embedding:
                embedding_manager_logger.warning("Could not generate embedding for query text in find_similar_texts.")
                return []

            candidate_embeddings = await self.embed_texts(candidate_texts)
            
            similarities = []
            for i, cand_text in enumerate(candidate_texts):
                cand_embedding = candidate_embeddings[i]
                if cand_embedding:
                    similarity_score = self._cosine_similarity_py(query_embedding, cand_embedding)
                    if similarity_score >= threshold:
                        similarities.append({
                            'text': cand_text,
                            'similarity': similarity_score,
                            'original_index': i
                        })
            
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            embedding_manager_logger.info(f"Found {len(similarities[:top_k])} similar texts.", parameters={'top_k': top_k, 'threshold': threshold})
            return similarities[:top_k]
            
        except Exception as e:
            embedding_manager_logger.error("Failed to find similar texts.", exception=e)
            return []
    
    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            cache_size = len(self._cache)
            
        cache_model_name = self._get_active_model_name_for_cache()
        cache_file = self.cache_dir / f"{cache_model_name.replace('/', '_')}_embedding_cache.pkl"
        cache_file_size = 0
        if cache_file.exists():
            try:
                cache_file_size = cache_file.stat().st_size
            except FileNotFoundError:
                pass
        
        stats = {
            "cache_enabled": self.cache_enabled,
            "in_memory_cache_items": cache_size,
            "disk_cache_file_size_bytes": cache_file_size,
            "cache_directory": str(self.cache_dir),
            "embedding_model_name": self._active_provider_model_name if self._initialized else self.default_model_name
        }
        embedding_manager_logger.info("Cache statistics retrieved.", parameters=stats)
        return stats
    
    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def clear_cache(self) -> None:
        """Clear the embedding cache from memory and disk."""
        embedding_manager_logger.info("Clearing embedding cache.")
        with self._cache_lock:
            self._cache.clear()
        
        cache_model_name = self._get_active_model_name_for_cache()
        cache_file = self.cache_dir / f"{cache_model_name.replace('/', '_')}_embedding_cache.pkl"
        if cache_file.exists():
            try:
                cache_file.unlink()
                embedding_manager_logger.info("Disk cache file deleted.", parameters={'cache_file': str(cache_file)})
            except OSError as e:
                embedding_manager_logger.error("Failed to delete disk cache file.", parameters={'cache_file': str(cache_file)}, exception=e)
        
        embedding_manager_logger.info("Embedding cache cleared successfully.")
    
    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def shutdown(self) -> None:
        """Shutdown the embedding manager, ensuring cache is saved."""
        embedding_manager_logger.info("Shutting down EmbeddingManager.")
        if self.cache_enabled and self._cache:
            await self._save_cache_async()
        
        self._initialized = False
        embedding_manager_logger.info("EmbeddingManager shutdown complete.")
    
    @detailed_log_function(LogCategory.VECTOR_STORE)
    async def health_check(self) -> Dict[str, Any]:
        """Check health of embedding manager."""
        if not self._initialized:
            return {"status": "uninitialized", "manager_name": "EmbeddingManager", "timestamp": datetime.now().isoformat()}
        
        embedding_manager_logger.debug("Performing EmbeddingManager health check.")
        try:
            test_embedding = await self.embed_text("health_check_string")
            embedding_works = bool(test_embedding) and len(test_embedding) > 0
            
            cache_stats = await self.get_cache_statistics()
            
            health_status = {
                "status": "healthy" if embedding_works else "degraded",
                "embedding_model_name": self._active_provider_model_name,
                "test_embedding_successful": embedding_works,
                "embedding_dimension": len(test_embedding) if embedding_works else "N/A",
                "cache_statistics": cache_stats,
                "timestamp": datetime.now().isoformat()
            }
            embedding_manager_logger.info("EmbeddingManager health check complete.", parameters=health_status)
            return health_status
        except Exception as e:
            embedding_manager_logger.error("EmbeddingManager health check failed.", exception=e)
            return {
                "status": "unhealthy",
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def get_service_status(self) -> Dict[str, Any]:
        return await self.health_check()

# Factory for service container
def create_embedding_manager(service_config: Optional[Dict[str, Any]] = None) -> EmbeddingManager:
    cfg = service_config.get("embedding_manager_config", {}) if service_config else {}
    return EmbeddingManager(
        embedding_provider=None,  # Allow service container to inject a pre-configured provider
        default_model_name=cfg.get("DEFAULT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        cache_enabled=cfg.get("CACHE_ENABLED", True),
        cache_dir_str=cfg.get("CACHE_DIR", "./storage/embeddings_cache_manager"),
        batch_size=cfg.get("BATCH_SIZE", 32),
        service_config=cfg
    )
    
```