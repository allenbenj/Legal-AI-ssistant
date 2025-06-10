"""Workflow orchestrator service.

Coordinates execution of document analysis workflows such as
:class:`RealTimeAnalysisWorkflow`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.document_utils import extract_text
from ..workflows.advanced_langgraph import build_advanced_legal_workflow
from ..api.websocket_manager import ConnectionManager

from ..core.detailed_logging import (
    get_detailed_logger,
    LogCategory,
    detailed_log_function,
)
from .realtime_analysis_workflow import RealTimeAnalysisWorkflow

wo_logger = get_detailed_logger("WorkflowOrchestrator", LogCategory.SYSTEM)


class WorkflowOrchestrator:
    """High level service to run document workflows."""

    @detailed_log_function(LogCategory.SYSTEM)
    def __init__(
        self,
        service_container,
        topic: str = "default",
        builder_topic: Optional[str] = None,
        workflow_config=None,
        **config: Any,
    ) -> None:
        """Initialize the orchestrator with optional workflow component config."""
        from .workflow_config import WorkflowConfig

        self.service_container = service_container
        self.config = config
        self.topic = topic
        self.builder_topic = builder_topic or topic
        self.graph_builder = build_advanced_legal_workflow
        self._graph = None
        self.websocket_manager: Optional[ConnectionManager] = None

        if workflow_config is None:
            workflow_config = config.get("workflow_config", WorkflowConfig())

        # RealTimeAnalysisWorkflow used for async document processing
        self.workflow = RealTimeAnalysisWorkflow(
            service_container,
            workflow_config=workflow_config,
            **config,
        )


        wo_logger.info("WorkflowOrchestrator initialized")

    async def _forward_progress(self, message: str, progress: float) -> None:
        """Forward workflow progress via WebSocket if manager available."""
        if not self.websocket_manager:
            return
        try:
            await self.websocket_manager.broadcast(
                f"workflow_progress_{self.topic}",
                {
                    "type": "processing_progress",
                    "message": message,
                    "progress": float(progress),
                },
            )
        except Exception as exc:  # pragma: no cover - network issues
            wo_logger.error(
                "Failed to broadcast progress update.", exception=exc
            )

    @detailed_log_function(LogCategory.SYSTEM)
    async def initialize_service(self) -> None:
        # Ensure container services are initialized before workflow runs
        if hasattr(self.service_container, "initialize_all_services"):
            await self.service_container.initialize_all_services()

        # Fetch websocket manager if available
        try:
            self.websocket_manager = await self.service_container.get_service(
                "websocket_manager"
            )
        except Exception:  # pragma: no cover - optional dependency
            self.websocket_manager = None

        await self.workflow.initialize()

        if self.websocket_manager:
            self.workflow.register_progress_callback(self._forward_progress)
        # lazily build graph for builder-based workflow
        if self._graph is None:
            self._graph = self.graph_builder(self.topic)

    def _create_builder_graph(self, topic: Optional[str] = None):
        """Return a LangGraph graph for the provided topic."""
        actual_topic = topic or self.builder_topic
        return build_advanced_legal_workflow(actual_topic)

    @detailed_log_function(LogCategory.SYSTEM)
    async def execute_workflow_instance(
        self,
        document_path_str: str,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Run the workflow and return the generated document ID."""

        # 1. Run the async real-time workflow
        result = await self.workflow.process_document_realtime(
            document_path=document_path_str,
            **(custom_metadata or {}),
        )

        # 2. Execute the builder based graph for additional processing
        if self._graph is None:
            self._graph = self.graph_builder(self.topic)
        text = extract_text(Path(document_path_str))
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._graph.run, text)

        return result.document_id

    @detailed_log_function(LogCategory.SYSTEM)
    async def execute_builder_workflow(
        self, document_path_str: str, topic: Optional[str] = None
    ) -> Any:
        """Run the LangGraph builder workflow on a document."""
        text = extract_text(Path(document_path_str))
        graph = self._create_builder_graph(topic)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, graph.run, text)


__all__ = ["WorkflowOrchestrator"]
