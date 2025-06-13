from __future__ import annotations

"""Service wrapper for keyword extraction utilities."""

from typing import List, Tuple

from ..core.detailed_logging import get_detailed_logger, LogCategory
from ..analytics.keyword_extractor import extract_keywords


class KeywordExtractionService:
    """Provide keyword extraction via TF-IDF ranking."""

    def __init__(self) -> None:
        self.logger = get_detailed_logger("KeywordExtractionService", LogCategory.SYSTEM)

    def extract(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return top ``top_k`` keywords and scores from ``text``."""
        self.logger.debug(
            "Extracting keywords",
            parameters={"length": len(text) if text else 0, "top_k": top_k},
        )
        return extract_keywords(text, top_k=top_k)
