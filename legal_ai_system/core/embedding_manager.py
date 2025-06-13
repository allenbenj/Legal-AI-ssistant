"""
Embedding Manager - Centralized Embedding Operations

This module provides the EmbeddingManager class for handling all embedding
operations across the legal AI system.
"""

import logging
import asyncio
import hashlib
import pickle
from typing import Dict, List, Any
from pathlib import Path
import threading

# Resolve the embedding client relative to this package so the module can be
# executed directly from the repository without installing ``legal_ai_system``.
from ..integration_ready.vector_store_enhanced import EmbeddingClient

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Centralized manager for embedding operations
    Handles caching, batching, and provider management
    """
    
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        cache_enabled: bool = True,
        cache_dir: str = "./storage/embeddings_cache",
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        
        self._embedding_client: EmbeddingClient | None = None
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        self._initialized = False
        
        logger.info(f"EmbeddingManager initialized with model: {model_name}")
    
    async def initialize(self) -> None:
        """Initialize the embedding manager"""
        if self._initialized:
            logger.warning("EmbeddingManager already initialized")
            return
        
        try:
            # Initialize the embedding client
            self._embedding_client = EmbeddingClient(model=self.model_name)
            
            # Create cache directory
            if self.cache_enabled:
                Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
                await self._load_cache()
            
            self._initialized = True
            logger.info("EmbeddingManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingManager: {e}")
            raise
    
    async def _load_cache(self) -> None:
        """Load embedding cache from disk"""
        cache_file = Path(self.cache_dir) / "embedding_cache.pkl"
        
        if cache_file.exists():
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._load_cache_sync, cache_file
                )
                logger.info(f"Loaded {len(self._cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self._cache = {}
    
    def _load_cache_sync(self, cache_file: Path) -> None:
        """Synchronous cache loading"""
        with open(cache_file, 'rb') as f:
            self._cache = pickle.load(f)
    
    async def _save_cache(self) -> None:
        """Save embedding cache to disk"""
        if not self.cache_enabled:
            return
        
        cache_file = Path(self.cache_dir) / "embedding_cache.pkl"
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._save_cache_sync, cache_file
            )
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def _save_cache_sync(self, cache_file: Path) -> None:
        """Synchronous cache saving"""
        with self._cache_lock:
            with open(cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Create a hash of the text and model name for caching
        content = f"{self.model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def embed_text(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        embeddings = await self.embed_texts([text])
        return embeddings[0] if embeddings else []
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        if not self._initialized:
            raise RuntimeError("EmbeddingManager not initialized")

        if self._embedding_client is None:
            raise RuntimeError("Embedding client not initialized")
        
        if not texts:
            return []
        
        try:
            # Check cache for existing embeddings
            cached_embeddings = {}
            uncached_texts = {}
            
            if self.cache_enabled:
                with self._cache_lock:
                    for i, text in enumerate(texts):
                        cache_key = self._get_cache_key(text)
                        if cache_key in self._cache:
                            cached_embeddings[i] = self._cache[cache_key]
                        else:
                            uncached_texts[i] = text
            else:
                uncached_texts = {i: text for i, text in enumerate(texts)}
            
            # Generate embeddings for uncached texts
            new_embeddings = {}
            if uncached_texts:
                uncached_list = list(uncached_texts.values())
                
                # Generate embeddings in batches
                for i in range(0, len(uncached_list), self.batch_size):
                    batch = uncached_list[i:i + self.batch_size]
                    batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                        None, self._embedding_client.embed, batch
                    )
                    
                    # Store results
                    for j, embedding in enumerate(batch_embeddings):
                        original_index = list(uncached_texts.keys())[i + j]
                        new_embeddings[original_index] = embedding.tolist()
                        
                        # Cache the embedding
                        if self.cache_enabled:
                            cache_key = self._get_cache_key(uncached_list[i + j])
                            with self._cache_lock:
                                self._cache[cache_key] = embedding.tolist()
            
            # Combine cached and new embeddings in original order
            all_embeddings = []
            for i in range(len(texts)):
                if i in cached_embeddings:
                    all_embeddings.append(cached_embeddings[i])
                elif i in new_embeddings:
                    all_embeddings.append(new_embeddings[i])
                else:
                    logger.error(f"Missing embedding for text at index {i}")
                    all_embeddings.append([])
            
            # Save cache periodically
            if self.cache_enabled and new_embeddings:
                await self._save_cache()
            
            logger.debug(f"Generated embeddings for {len(texts)} texts ({len(new_embeddings)} new, {len(cached_embeddings)} cached)")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    async def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add embeddings to document dictionaries"""
        if not self._initialized:
            raise RuntimeError("EmbeddingManager not initialized")
        
        try:
            # Extract text content from documents
            texts = []
            for doc in documents:
                text = doc.get('text', doc.get('content', ''))
                if not text:
                    logger.warning(f"Document missing text content: {doc.get('id', 'unknown')}")
                    text = ""
                texts.append(text)
            
            # Generate embeddings
            embeddings = await self.embed_texts(texts)
            
            # Add embeddings to documents
            enhanced_documents = []
            for i, doc in enumerate(documents):
                enhanced_doc = doc.copy()
                enhanced_doc['embedding'] = embeddings[i] if i < len(embeddings) else []
                enhanced_documents.append(enhanced_doc)
            
            logger.info(f"Added embeddings to {len(enhanced_documents)} documents")
            return enhanced_documents
            
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise
    
    async def compute_similarity(
        self,
        text1: str,
        text2: str,
        similarity_type: str = "cosine"
    ) -> float:
        """Compute similarity between two texts"""
        if not self._initialized:
            raise RuntimeError("EmbeddingManager not initialized")
        
        try:
            embeddings = await self.embed_texts([text1, text2])
            
            if len(embeddings) != 2 or not embeddings[0] or not embeddings[1]:
                return 0.0
            
            if similarity_type == "cosine":
                return self._cosine_similarity(embeddings[0], embeddings[1])
            elif similarity_type == "euclidean":
                return self._euclidean_similarity(embeddings[0], embeddings[1])
            else:
                raise ValueError(f"Unsupported similarity type: {similarity_type}")
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _euclidean_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute normalized euclidean similarity between two vectors"""
        import math
        
        # Compute euclidean distance
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
        
        # Convert to similarity (closer to 1 means more similar)
        return 1.0 / (1.0 + distance)
    
    async def find_similar_texts(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Find most similar texts to a query"""
        if not self._initialized:
            raise RuntimeError("EmbeddingManager not initialized")
        
        try:
            # Generate embeddings for query and candidates
            all_texts = [query_text] + candidate_texts
            embeddings = await self.embed_texts(all_texts)
            
            if not embeddings or not embeddings[0]:
                return []
            
            query_embedding = embeddings[0]
            candidate_embeddings = embeddings[1:]
            
            # Compute similarities
            similarities = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                if candidate_embedding:
                    similarity = self._cosine_similarity(query_embedding, candidate_embedding)
                    if similarity >= threshold:
                        similarities.append({
                            'index': i,
                            'text': candidate_texts[i],
                            'similarity': similarity
                        })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find similar texts: {e}")
            return []
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._cache_lock:
            cache_size = len(self._cache)
            
        cache_file = Path(self.cache_dir) / "embedding_cache.pkl"
        cache_file_size = 0
        if cache_file.exists():
            cache_file_size = cache_file.stat().st_size
        
        return {
            "cache_enabled": self.cache_enabled,
            "cache_size": cache_size,
            "cache_file_size_bytes": cache_file_size,
            "cache_dir": self.cache_dir,
            "model_name": self.model_name
        }
    
    async def clear_cache(self) -> None:
        """Clear the embedding cache"""
        with self._cache_lock:
            self._cache.clear()
        
        cache_file = Path(self.cache_dir) / "embedding_cache.pkl"
        if cache_file.exists():
            cache_file.unlink()
        
        logger.info("Embedding cache cleared")
    
    async def shutdown(self) -> None:
        """Shutdown the embedding manager"""
        if self.cache_enabled and self._cache:
            await self._save_cache()
        
        self._initialized = False
        logger.info("EmbeddingManager shut down")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of embedding manager"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        try:
            # Test embedding generation
            test_embedding = await self.embed_text("test")
            
            cache_stats = await self.get_cache_statistics()
            
            return {
                "status": "healthy",
                "model_name": self.model_name,
                "embedding_dimension": len(test_embedding) if test_embedding else 0,
                "cache_statistics": cache_stats
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
