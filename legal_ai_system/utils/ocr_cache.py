from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Optional

try:
    from diskcache import Cache
except Exception:  # pragma: no cover - diskcache may be missing in some envs
    Cache = None  # type: ignore


CACHE_DIR = Path(os.environ.get("OCR_CACHE_DIR", Path.home() / ".ocr_cache"))
_cache: Optional[Cache] = Cache(str(CACHE_DIR)) if Cache else None


def compute_file_hash(path: Path) -> str:
    """Return SHA256 hash of file contents."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _make_key(path: Path, page: int, file_hash: str) -> str:
    return f"{file_hash}:{page}:{path.name}"


def get(path: Path, page: int = 0, *, file_hash: Optional[str] = None) -> Optional[str]:
    """Return cached OCR text for a page if available."""
    if not _cache:
        return None
    file_hash = file_hash or compute_file_hash(path)
    return _cache.get(_make_key(path, page, file_hash))


def set(path: Path, page: int, text: str, *, file_hash: Optional[str] = None) -> None:
    """Store OCR text for a page."""
    if not _cache:
        return
    file_hash = file_hash or compute_file_hash(path)
    _cache.set(_make_key(path, page, file_hash), text)

