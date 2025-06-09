"""LangGraph nodes that interface with LLM providers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

class BaseNode:  # pragma: no cover - fallback when langgraph is missing
    """Minimal stand-in for :class:`langgraph.graph.BaseNode`."""

    pass


try:  # pragma: no cover - optional dependency
    from langgraph.graph import BaseNode as LangGraphBaseNode
    BaseNode = LangGraphBaseNode  # type: ignore[assignment]
except Exception:  # ImportError or other issues if langgraph not installed
    pass


if TYPE_CHECKING:  # pragma: no cover - hint for type checkers
    from langgraph.graph import BaseNode as LangGraphBaseNode
from ..core.unified_services import get_service_container, register_core_services
from ..services.integration_service import (
    LegalAIIntegrationService,
    create_integration_service,
)


class AnalysisNode(BaseNode):
    """Run analysis on provided text using the integration service."""

    def __init__(self, topic: str):
        self.topic = topic
        if LegalAIIntegrationService:
            register_core_services()
            container = get_service_container()
            self.service = create_integration_service(container)
        else:
            self.service = None

    def __call__(self, input_text: str) -> str:
        if not self.service or not hasattr(self.service, "analyze_text"):
            return f"Analysis not available for {self.topic}: {input_text[:30]}"
        result = self.service.analyze_text(input_text, topic=self.topic)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        return result


class SummaryNode(BaseNode):
    """Summarize text using the integration service."""

    def __init__(self):
        if LegalAIIntegrationService:
            register_core_services()
            container = get_service_container()
            self.service = create_integration_service(container)
        else:
            self.service = None

    def __call__(self, input_text: str) -> str:
        if not self.service or not hasattr(self.service, "summarize_text"):
            return input_text[:200]
        result = self.service.summarize_text(input_text)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        return result


__all__ = ["AnalysisNode", "SummaryNode"]
