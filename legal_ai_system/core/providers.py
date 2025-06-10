# legal_ai_system/core/embeddings/providers.py
"""
Embedding Providers for the Legal AI System.
Defines the interface and concrete implementations for generating text embeddings.
"""
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np  # type: ignore

from legal_ai_system.core.detailed_logging import (
    LogCategory,
    detailed_log_function,
    get_detailed_logger,
)
from legal_ai_system.core.unified_exceptions import (
    ConfigurationError,
    ThirdPartyError,
)


class EmbeddingProviderVS(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.dimension: Optional[int] = None
        self.logger = embedding_provider_logger.getChild(self.__class__.__name__)

    @abstractmethod
    async def initialize(self):
        """Initialize the embedding model and resources."""
        self.logger.info(
            "Initializing embedding provider.",
            parameters={"model_name": self.model_name},
        )
        pass

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the embedding provider."""
        # Default implementation, can be overridden
        is_init = self.dimension is not None
        status = {
            "provider_name": self.__class__.__name__,
            "model_name": self.model_name,
            "status": "healthy" if is_init else "uninitialized",
            "dimension": self.dimension,
        }
        self.logger.debug("Health check performed.", parameters=status)
        return status


class SentenceTransformerEmbeddingProvider(EmbeddingProviderVS):
    """Embedding provider using SentenceTransformers library."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.model: Optional[Any] = None  # For SentenceTransformer model object

    @detailed_log_function(LogCategory.LLM)
    async def initialize(self):
        await super().initialize()  # Call ABC's initialize logging
        self.logger.info("Initializing SentenceTransformer model.")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            # This is a synchronous call, run in executor if it's slow
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, SentenceTransformer, self.model_name
            )
            if not self.model:
                raise RuntimeError("SentenceTransformer model loading returned None.")
            self.dimension = self.model.get_sentence_embedding_dimension()  # type: ignore
            self.logger.info(
                "SentenceTransformer model loaded successfully.",
                parameters={"model": self.model_name, "dimension": self.dimension},
            )
        except ImportError:
            self.logger.critical(
                "SentenceTransformers library not found. Please install it: pip install sentence-transformers"
            )
            raise ConfigurationError(
                "SentenceTransformers library not installed.",
                config_key="embedding_provider_sentencetransformers",
            )
        except Exception as e:
            self.logger.error("Failed to load SentenceTransformer model.", exception=e)
            raise ThirdPartyError(
                f"Failed to load SentenceTransformer model {self.model_name}",
                provider_name="SentenceTransformers",
                cause=e,
            )

    @detailed_log_function(LogCategory.LLM)
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not self.model or self.dimension is None:
            self.logger.error(
                "SentenceTransformer model not initialized before embedding."
            )
            raise RuntimeError(
                "SentenceTransformer model not initialized. Call initialize() first."
            )

        self.logger.debug(f"Embedding {len(texts)} texts with SentenceTransformer.")
        start_time = time.perf_counter()

        loop = asyncio.get_event_loop()
        # The encode method can be CPU intensive
        embeddings_np_arrays = await loop.run_in_executor(None, self.model.encode, texts, {"show_progress_bar": False})  # type: ignore

        embeddings_normalized: List[List[float]] = []
        for emb_np in embeddings_np_arrays:
            if not isinstance(emb_np, np.ndarray):  # Ensure it's a numpy array
                self.logger.warning(
                    "Embedding provider returned non-numpy array. Skipping normalization.",
                    parameters={"type": type(emb_np).__name__},
                )
                embeddings_normalized.append(
                    list(emb_np) if isinstance(emb_np, list) else []
                )  # Convert if list, else empty
                continue

            norm = np.linalg.norm(emb_np)
            normalized_emb = (emb_np / norm if norm > 0 else emb_np).tolist()
            embeddings_normalized.append(normalized_emb)

        duration = time.perf_counter() - start_time
        self.logger.debug(
            f"Generated {len(embeddings_normalized)} embeddings.",
            parameters={"duration_sec": duration, "num_texts": len(texts)},
        )
        return embeddings_normalized


# Add other provider implementations here if needed (e.g., OllamaEmbeddingProvider, OpenAIEmbeddingProvider)


# Factory function for service container
def create_embedding_provider(config: Dict[str, Any]) -> EmbeddingProviderVS:
    provider_type = config.get(
        "embedding_provider_type", "sentence_transformer"
    ).lower()
    model_name = config.get(
        "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_provider_logger.info(
        "Creating embedding provider via factory.",
        parameters={"type": provider_type, "model": model_name},
    )

    if provider_type == "sentence_transformer":
        return SentenceTransformerEmbeddingProvider(model_name=model_name)
    # Add other providers:
    # elif provider_type == "ollama":
    #     return OllamaEmbeddingProvider(model_name=model_name, ollama_host=config.get("ollama_host"))
    # elif provider_type == "openai":
    #     return OpenAIEmbeddingProvider(model_name=model_name, api_key=config.get("openai_api_key"))
    else:
        msg = f"Unsupported embedding provider type: {provider_type}"
        embedding_provider_logger.error(msg)
        raise ConfigurationError(msg, config_key="embedding_provider_type")
