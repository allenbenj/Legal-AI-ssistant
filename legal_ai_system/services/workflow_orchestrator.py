"""Workflow orchestrator service.

Coordinates execution of document analysis workflows such as
:class:`RealTimeAnalysisWorkflow`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional
import time

from ..utils.document_utils import extract_text
from ..workflows.langgraph_setup import build_graph
from ..workflows.case_workflow_state import CaseWorkflowState

from ..core.detailed_logging import (
    get_detailed_logger,
    LogCategory,
    detailed_log_function,
)
from .realtime_analysis_workflow import RealTimeAnalysisWorkflow
from .metrics_exporter import metrics_exporter

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

        self.graph_builder = build_graph
        self._graph = None

        if workflow_config is None:
            workflow_config = config.get("workflow_config", WorkflowConfig())

        # RealTimeAnalysisWorkflow used for async document processing
        task_queue = config.get("task_queue") or service_container._services.get(
            "task_queue"
        )
        self.workflow = RealTimeAnalysisWorkflow(
            service_container,
            workflow_config=workflow_config,
            task_queue=task_queue,
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

        await self.workflow.initialize()

        if self.websocket_manager:
            self.workflow.register_progress_callback(self._forward_progress)
        # lazily build graph for builder-based workflow
        if self._graph is None:
            self._graph = self.graph_builder(self.topic)

        # Setup progress forwarding if connection manager available
        if ConnectionManager:
            try:
                self.connection_manager = await self.service_container.get_service(
                    "connection_manager"
                )
            except Exception:
                self.connection_manager = None

        if self.connection_manager:
            async def _progress_cb(message: str, progress: float) -> None:
                await self.connection_manager.broadcast(
                    "workflow_progress",
                    {
                        "type": "workflow_progress",
                        "message": message,
                        "progress": float(progress),
                    },
                )

            self.workflow.register_progress_callback(_progress_cb)

    def _create_builder_graph(self, topic: Optional[str] = None):
        """Return a LangGraph graph for the provided topic."""
        actual_topic = topic or self.builder_topic
        return self.graph_builder(actual_topic)

    @detailed_log_function(LogCategory.SYSTEM)
    async def execute_workflow_instance(
        self,
        document_path_str: str,
        custom_metadata: Optional[Dict[str, Any]] = None,
        case_state: "CaseWorkflowState" | None = None,
    ) -> str:
        """Run the workflow and return the generated document ID."""

        start_time = time.time()
        metrics = metrics_exporter

        try:
            # 1. Run the async real-time workflow
            result = await self.workflow.process_document_realtime(
                document_path=document_path_str,
                **(custom_metadata or {}),
            )

            if self._graph is None:
                self._graph = self.graph_builder(self.topic)

            # Prepare text to feed the graph. When a case state object is provided
            # we keep a cumulative context so the graph can reason over multiple
            # documents from the same case.
            text = extract_text(Path(document_path_str))
            if case_state is not None:
                case_state.process_new_document(result.document_id, text)
                graph_input = case_state.get_case_context()
            else:
                graph_input = text

            # 2. Execute the builder based graph for additional processing
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._graph.run, graph_input)

            if case_state is not None:
                case_state.update_case_state({"last_processed": result.document_id})

            return result.document_id
        except Exception as exc:  # pragma: no cover - propagate after logging
            if metrics:
                metrics.inc_workflow_error()
            wo_logger.error("Workflow execution failed", exception=exc)
            raise
        finally:
            if metrics:
                metrics.observe_workflow_time(time.time() - start_time)

    @detailed_log_function(LogCategory.SYSTEM)
    async def execute_builder_workflow(
        self, document_path_str: str, topic: Optional[str] = None
    ) -> Any:
        """Run the LangGraph builder workflow on a document."""
        text = extract_text(Path(document_path_str))
        graph = self._create_builder_graph(topic)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, graph.run, text)


__all__ = ["WorkflowOrchestrator", "build_graph"]
