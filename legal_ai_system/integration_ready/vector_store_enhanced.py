# Auto-generated compatibility wrappers for legacy imports.
# This module provides minimal implementations of ``MemoryStore`` and
# ``EmbeddingClient`` so existing code referencing
# ``legal_ai_system.integration_ready.vector_store_enhanced`` continues to run.

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


class MemoryStore:
    """Lightweight placeholder used by :class:`MemoryManager`."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:  # pragma: no cover - simple wrapper
        """Initialize backing storage if required."""
        self.db_path.touch(exist_ok=True)

    async def close(self) -> None:  # pragma: no cover - placeholder
        """Close any open resources."""
        return None


class EmbeddingClient:
    """Simple embedding client using SentenceTransformers if available."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:  # pragma: no cover - optional dependency
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._model = None

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        self._load_model()
        if self._model is not None:
            embeddings = self._model.encode(list(texts), show_progress_bar=False)
            return [list(map(float, emb)) for emb in embeddings]

        # Fallback: very small deterministic embedding
        return [[float(ord(ch)) / 255.0 for ch in text][:32] for text in texts]
