"""LangGraph nodes that interface with LLM providers."""

from __future__ import annotations

from typing import Any

from langgraph.graph import BaseNode
from ..services.integration_service import (
    LegalAIIntegrationService,
    create_integration_service,
)
from ..core.unified_services import get_service_container, register_core_services


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
        return self.service.analyze_text(input_text, topic=self.topic)


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
        return self.service.summarize_text(input_text)

__all__ = ["AnalysisNode", "SummaryNode"]
