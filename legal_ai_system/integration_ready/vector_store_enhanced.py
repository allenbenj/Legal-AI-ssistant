import logging
from pathlib import Path
from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Lightweight embedding client using SentenceTransformers if available."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model
        self.model = None
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model)
            except Exception as exc:  # pragma: no cover - initialization can fail
                logger.warning("Failed to load SentenceTransformer model; falling back to simple embeddings", exc_info=exc)
        self.dimension = getattr(self.model, "get_sentence_embedding_dimension", lambda: 128)()

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Return embeddings for a batch of texts."""
        if self.model is not None:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return [np.asarray(e, dtype=float) for e in embeddings]

        # Fallback: simple character code based embeddings
        result = []
        for text in texts:
            arr = np.zeros(self.dimension, dtype=float)
            for i, char in enumerate(text[: self.dimension]):
                arr[i] = ord(char) / 255.0
            result.append(arr)
        return result


class MemoryStore:
    """Minimal placeholder memory store for integration."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        try:
            import sqlite3

            self.connection = sqlite3.connect(self.db_path)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Failed to initialize SQLite connection", exc_info=exc)

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
