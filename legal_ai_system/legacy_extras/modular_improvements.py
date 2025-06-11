"""Backward-compatible utilities used by some legacy agents."""

from __future__ import annotations

from typing import Any, Dict, Tuple


class ProcessingCache:
    """Simple asynchronous in-memory cache used for intermediate results."""

    def __init__(self) -> None:
        self._store: Dict[Tuple[str, str], Any] = {}

    async def get(self, document_id: str, key: str) -> Any:
        """Return a cached item if present."""
        return self._store.get((document_id, key))

    async def set(self, document_id: str, key: str, value: Any) -> None:
        """Store an item in the cache."""
        self._store[(document_id, key)] = value

    async def clear(self) -> None:
        """Clear all cached data."""
        self._store.clear()
