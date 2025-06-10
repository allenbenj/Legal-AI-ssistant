"""Generic error-handling workflow node."""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict

logger = logging.getLogger(__name__)


class LegalErrorHandlingNode:
    """Attempt processing with fallback and escalation."""

    def __init__(
        self,
        primary_processor: Callable[[Any], Awaitable[Dict[str, Any]]],
        ensemble_analyzer: Callable[[Any], Awaitable[Dict[str, Any]]],
        escalate: Callable[[Any, Exception | None], Awaitable[None]] | None = None,
        *,
        confidence_key: str = "confidence",
        threshold: float = 0.8,
    ) -> None:
        self.primary_processor = primary_processor
        self.ensemble_analyzer = ensemble_analyzer
        self.escalate = escalate or (lambda *_args: logger.error("Escalation triggered"))
        self.confidence_key = confidence_key
        self.threshold = threshold

    async def __call__(self, data: Any) -> Dict[str, Any]:
        try:
            result = await self.primary_processor(data)
            if result.get(self.confidence_key, 0.0) >= self.threshold:
                return result
            logger.info("Low confidence from primary processor; running ensemble analysis")
            result = await self.ensemble_analyzer(data)
            if result.get(self.confidence_key, 0.0) >= self.threshold:
                return result
            await self.escalate(data, None)
            result["escalated"] = True
            return result
        except Exception as exc:  # pragma: no cover - unexpected failure path
            logger.error("Primary processor failed: %s", exc)
            try:
                result = await self.ensemble_analyzer(data)
                if result.get(self.confidence_key, 0.0) >= self.threshold:
                    return result
                await self.escalate(data, exc)
                result["escalated"] = True
                return result
            except Exception as ensemble_exc:  # pragma: no cover - ensemble failure
                logger.critical("Ensemble analysis failed: %s", ensemble_exc)
                await self.escalate(data, ensemble_exc)
                return {"error": str(ensemble_exc), "escalated": True}


__all__ = ["LegalErrorHandlingNode"]
