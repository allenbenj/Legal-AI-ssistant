"""Expose default configuration constants at the package root."""

from legal_ai_system.config import (
    BASE_DIR,
    CACHE_DIR,
    DATA_DIR,
    DEFAULT_CONFIG,
    DOCS_DIR,
    EMBEDDINGS_DIR,
)

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "DOCS_DIR",
    "CACHE_DIR",
    "EMBEDDINGS_DIR",
    "DEFAULT_CONFIG",
]
