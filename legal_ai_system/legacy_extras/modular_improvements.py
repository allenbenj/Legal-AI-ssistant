import json
from pathlib import Path
from typing import Any, Optional

class ProcessingCache:
    """Simple file-based cache for processed results."""

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.base_dir = Path(base_dir or Path(__file__).resolve().parents[2] / "storage" / "cache")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def get(self, document_id: str, cache_key: str) -> Optional[Any]:
        """Retrieve cached data if available."""
        cache_file = self.base_dir / f"{document_id}_{cache_key}.json"
        if cache_file.exists():
            try:
                with cache_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    async def set(self, document_id: str, cache_key: str, value: Any) -> None:
        """Store processed data in the cache."""
        cache_file = self.base_dir / f"{document_id}_{cache_key}.json"
        try:
            with cache_file.open("w", encoding="utf-8") as f:
                json.dump(value, f, default=str)
        except Exception:
            pass

