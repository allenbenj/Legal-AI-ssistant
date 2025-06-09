"""Central configuration for default paths and settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Default storage directories
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "documents"
CACHE_DIR = DATA_DIR / "cache"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Ensure directories exist
for _dir in [DATA_DIR, DOCS_DIR, CACHE_DIR, EMBEDDINGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# Example configuration dictionary for service initialization
DEFAULT_CONFIG: Dict[str, Any] = {
    "vector_store_path": EMBEDDINGS_DIR / "vector_store.faiss",
    "memory_db_path": DATA_DIR / "memory.db",
}

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "DOCS_DIR",
    "CACHE_DIR",
    "EMBEDDINGS_DIR",
    "DEFAULT_CONFIG",
]
