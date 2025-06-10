"""Workflow orchestrator service.

Coordinates execution of document analysis workflows such as
:class:`RealTimeAnalysisWorkflow`.
"""

from typing import Any, Dict, Optional

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
    def __init__(self, service_container, workflow_config=None, **config: Any) -> None:
        """Initialize the orchestrator with optional workflow component config."""
        from ..config.workflow_config import WorkflowConfig

        self.service_container = service_container
        self.config = config

        if workflow_config is None:
            workflow_config = config.get("workflow_config", WorkflowConfig())

        self.workflow = RealTimeAnalysisWorkflow(
            service_container,
            workflow_config=workflow_config,
            **config,
        )
        wo_logger.info("WorkflowOrchestrator initialized")

    @detailed_log_function(LogCategory.SYSTEM)
    async def initialize_service(self) -> None:
        await self.workflow.initialize()

    @detailed_log_function(LogCategory.SYSTEM)
    async def execute_workflow_instance(
        self,
        document_path_str: str,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Run the workflow and return the generated document ID."""
        result = await self.workflow.process_document_realtime(
            document_path=document_path_str,
            **(custom_metadata or {}),
        )
        return result.document_id
