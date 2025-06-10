"""Workflow node for robust error handling and fallback logic."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from ...workflow_engine.types import LegalWorkflowNode


class LegalErrorHandlingNode:
    """Attempt primary processing and fallback to ensemble analysis.

    If both results have low confidence or an exception occurs, escalate via
    the provided callback or raise a ``RuntimeError``.
    """

    def __init__(
        self,
        primary_processor: LegalWorkflowNode,
        ensemble_analyzer: LegalWorkflowNode,
        *,
        confidence_threshold: float = 0.8,
        escalate: Optional[LegalWorkflowNode] = None,
    ) -> None:
        self.primary_processor = primary_processor
        self.ensemble_analyzer = ensemble_analyzer
        self.confidence_threshold = confidence_threshold
        self.escalate = escalate

    async def __call__(self, data: Any) -> Any:
        try:
            result = await self.primary_processor(data)
            if self._is_confident(result):
                return result
        except Exception as exc:
            return await self._escalate({"input": data, "error": exc})

        try:
            fallback = await self.ensemble_analyzer(data)
            if self._is_confident(fallback):
                return fallback
            return await self._escalate({"input": data, "result": fallback})
        except Exception as exc:
            return await self._escalate({"input": data, "error": exc})

    def _is_confident(self, result: Any) -> bool:
        confidence = None
        if isinstance(result, dict):
            confidence = result.get("confidence")
        else:
            confidence = getattr(result, "confidence", None)
        return confidence is not None and confidence >= self.confidence_threshold

    async def _escalate(self, info: Any) -> Any:
        if self.escalate is not None:
            return await self.escalate(info)
        raise RuntimeError("Escalation required")


__all__ = ["LegalErrorHandlingNode"]
