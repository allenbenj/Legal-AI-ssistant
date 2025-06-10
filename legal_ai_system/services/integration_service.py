# legal_ai_system/services/integration_service.py
"""
Integration Service for Legal AI System.

Provides a higher-level service facade that uses the ServiceContainer
to interact with various backend components (agents, managers, workflows).
This service is typically used by the API layer (e.g., FastAPI).
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

# import logging # Replaced by detailed_logging
from typing import Any, Dict, Optional, Tuple

from legal_ai_system.services.memory_manager import MemoryManager
from legal_ai_system.core.enhanced_persistence import EnhancedPersistenceManager

from legal_ai_system.core.detailed_logging import (
    LogCategory,
    detailed_log_function,
    get_detailed_logger,
)
from legal_ai_system.core.unified_exceptions import (
    ConfigurationError,
    ServiceLayerError,
)
from legal_ai_system.services.security_manager import (
    SecurityManager,
)
from legal_ai_system.services.security_manager import User as AuthUser
from legal_ai_system.services.service_container import ServiceContainer
from legal_ai_system.services.workflow_orchestrator import WorkflowOrchestrator

# Initialize logger for this module
integration_service_logger = get_detailed_logger("IntegrationService", LogCategory.API)


class LegalAIIntegrationService:
    """
    Integration service bridging API layer (e.g., FastAPI) and the core Legal AI System components.
    It uses the ServiceContainer for service discovery and orchestration.
    """

    @detailed_log_function(LogCategory.API)
    def __init__(self, service_container: ServiceContainer):
        integration_service_logger.info("Initializing LegalAIIntegrationService.")
        if not service_container:
            msg = "ServiceContainer must be provided to LegalAIIntegrationService."
            integration_service_logger.critical(msg)
            raise ConfigurationError(msg)

        self.service_container = service_container
        # Cached service references (may be None if not yet registered)
        self.security_manager: Optional[SecurityManager] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.workflow_orchestrator: Optional[WorkflowOrchestrator] = None
        self.llm_manager: Optional[Any] = None
        self.persistence_manager: Optional[EnhancedPersistenceManager] = None
        self.legal_reasoning_engine: Optional[Any] = None

        # Example: self.realtime_workflow: Optional[RealTimeAnalysisWorkflow] = None

        integration_service_logger.info(
            "LegalAIIntegrationService initialized successfully."
        )

    async def _save_uploaded_file(
        self, file_content: bytes, filename: str
    ) -> Tuple[Path, str]:
        """Save uploaded bytes to disk and return the resulting path."""
        upload_dir = Path("./storage/documents/uploads_service")
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_filename = "".join(
            c if c.isalnum() or c in [".", "-", "_"] else "_" for c in filename
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{safe_filename}"
        file_path = upload_dir / unique_filename

        try:
            with open(file_path, "wb") as f:
                f.write(file_content)
            integration_service_logger.info(
                "File saved to temporary upload location.",
                parameters={"path": str(file_path)},
            )
        except Exception as e:  # pragma: no cover - file write error
            integration_service_logger.error(
                "Failed to save uploaded file.",
                parameters={"filename": filename},
                exception=e,
            )
            raise ServiceLayerError("Failed to save uploaded file.") from e

        return file_path, unique_filename

    async def _create_document_metadata(
        self,
        file_path: Path,
        filename: str,
        user: AuthUser,
        options: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Persist document metadata and create session."""
        try:
            if not self.memory_manager:
                self.memory_manager = await self.service_container.get_service(
                    "memory_manager"
                )
            document_id = f"doc_serv_{uuid.uuid4().hex}"
            workflow_metadata = {
                "document_id": document_id,
                "original_filename": filename,
                "user_id": user.user_id,
                "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_options": options,
            }

            await self.create_document_record(
                document_id,
                filename,
                file_path,
                user.user_id,
                options,
            )

            if self.memory_manager:
                await self.memory_manager.create_session(
                    session_id=document_id,
                    session_name=filename,
                    metadata=workflow_metadata,
                )
            else:
                integration_service_logger.warning(
                    "MemoryManager unavailable; metadata not persisted"
                )
            return document_id, workflow_metadata
        except Exception as e:
            integration_service_logger.error(
                "Failed to create document metadata.",
                parameters={"filename": filename},
                exception=e,
            )
            raise ServiceLayerError("Failed to create document metadata.") from e

    async def _launch_workflow(
        self, file_path: Path, workflow_metadata: Dict[str, Any]
    ) -> None:
        """Start document workflow execution."""
        try:
            orchestrator = self.workflow_orchestrator
            if not orchestrator:
                orchestrator = await self.service_container.get_service(
                    "workflow_orchestrator"
                )
                self.workflow_orchestrator = orchestrator
            if not orchestrator:
                raise ServiceLayerError("Workflow orchestrator service not available.")

            self.service_container.add_background_task(
                orchestrator.execute_workflow_instance(
                    document_path_str=str(file_path),
                    custom_metadata=workflow_metadata,
                )
            )

            integration_service_logger.info(
                "Document processing initiated via workflow.",
                parameters={
                    "doc_id": workflow_metadata.get("document_id"),
                    "workflow_service": type(orchestrator).__name__,
                },
            )
        except ServiceLayerError:
            raise
        except Exception as e:
            integration_service_logger.error(
                "Failed to launch workflow.",
                parameters={"document_id": workflow_metadata.get("document_id")},
                exception=e,
            )
            raise ServiceLayerError("Failed to launch workflow.") from e

    async def create_document_record(
        self,
        document_id: str,
        filename: str,
        file_path: Path,
        user_id: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist metadata for an uploaded document.

        Parameters
        ----------
        document_id : str
            Identifier assigned to the uploaded document.
        filename : str
            Original filename provided by the client.
        file_path : Path
            Location on disk where the file was stored.
        user_id : str
            ID of the authenticated user uploading the file.
        options : Dict[str, Any], optional
            Additional metadata provided with the upload.
        """
        if not self.persistence_manager:
            self.persistence_manager = await self.service_container.get_service(
                "persistence_manager"
            )
        persistence = self.persistence_manager
        if not persistence:
            integration_service_logger.warning(
                "PersistenceManager unavailable; document record not stored"
            )
            return

        file_type = file_path.suffix.lstrip(".")
        try:
            file_size = file_path.stat().st_size
            file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        except Exception as e:  # pragma: no cover - file stat error
            integration_service_logger.error(
                "Failed to compute file metadata.", exception=e
            )
            file_size = 0
            file_hash = None

        metadata = dict(options or {})
        metadata.setdefault("user_id", user_id)

        try:
            async with persistence.connection_pool.get_pg_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO documents (
                        document_id, filename, file_path, file_size,
                        file_type, file_hash, processing_status,
                        created_at, updated_at, source, custom_metadata
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6,
                        'pending', NOW(), NOW(), $7, $8
                    )
                    ON CONFLICT (document_id) DO NOTHING
                    """,
                    document_id,
                    filename,
                    str(file_path),
                    file_size,
                    file_type,
                    file_hash,
                    "upload",
                    json.dumps(metadata),
                )
            integration_service_logger.debug(
                "Document record created.", parameters={"document_id": document_id}
            )
        except Exception as e:  # pragma: no cover - db failure
            integration_service_logger.error(
                "Failed to store document record.",
                parameters={"document_id": document_id},
                exception=e,
            )

    @detailed_log_function(LogCategory.API)
    async def handle_document_upload(
        self,
        file_content: bytes,
        filename: str,
        user: AuthUser,  # Pass authenticated user object
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Handle a document upload and start background processing.

        Returns a dictionary with the following keys::

            {
                "document_id": str,
                "filename": str,
                "size_bytes": int,
                "status": str,
                "message": str,
            }

        Raises:
            ServiceLayerError: if security checks, storage or workflow
            initialization fails.
        """
        integration_service_logger.info(
            "Handling document upload.",
            parameters={"filename": filename, "user_id": user.user_id},
        )
        options = options or {}

        if not self.security_manager:
            self.security_manager = await self.service_container.get_service(
                "security_manager"
            )
        if not self.security_manager:
            raise ServiceLayerError(
                "SecurityManager not available for document upload."
            )

        try:
            file_path, unique_filename = await self._save_uploaded_file(
                file_content, filename
            )
            document_id, workflow_metadata = await self._create_document_metadata(
                file_path, filename, user, options
            )
            await self._launch_workflow(file_path, workflow_metadata)

            return {
                "document_id": document_id,
                "filename": unique_filename,
                "size_bytes": len(file_content),
                "status": "processing_initiated",
                "message": "Document accepted and processing started.",
            }

        except ServiceLayerError:
            raise
        except Exception as e:
            integration_service_logger.error(
                "Failed to handle document upload and initiate processing.",
                parameters={"filename": filename},
                exception=e,
            )
            raise ServiceLayerError(
                f"Document upload handling failed: {str(e)}", cause=e
            )

    @detailed_log_function(LogCategory.API)
    async def get_document_analysis_status(
        self, document_id: str, user: AuthUser
    ) -> Dict[str, Any]:
        """Retrieves the processing status and summary for a document."""
        integration_service_logger.debug(
            "Fetching document analysis status.",
            parameters={"doc_id": document_id, "user_id": user.user_id},
        )
        # This would query a workflow state manager or document metadata store
        # Example:
        # workflow_state_manager = self.service_container.get_service("workflow_state_manager")
        # status = await workflow_state_manager.get_status(document_id)
        # For now, mock:
        if document_id == "doc_serv_test123":
            return {
                "document_id": document_id,
                "status": "completed",
                "progress": 1.0,
                "stage": "Done",
                "summary": {"entities_found": 10, "violations_detected": 1},
            }
        return {
            "document_id": document_id,
            "status": "processing",
            "progress": 0.5,
            "stage": "entity_extraction",
        }

    @detailed_log_function(LogCategory.API)
    async def analyze_text(self, text: str, topic: str = "general") -> str:
        """Analyze text using the configured LLM manager."""
        if not self.llm_manager and self.service_container:
            try:
                self.llm_manager = await self.service_container.get_service(
                    "llm_manager"
                )
            except Exception as e:  # pragma: no cover - service retrieval issue
                integration_service_logger.warning(
                    "LLMManager unavailable for analysis", exception=e
                )
        llm_manager = self.llm_manager
        if not llm_manager:
            return text
        prompt = f"Analyze the following text about {topic}:\n{text}"
        try:
            response = await llm_manager.complete(prompt)
            return response.content
        except Exception as e:  # pragma: no cover - LLM failure
            integration_service_logger.error("LLM analysis failed", exception=e)
            return text

    @detailed_log_function(LogCategory.API)
    async def summarize_text(self, text: str, max_tokens: int = 200) -> str:
        """Summarize text using the LLM manager."""
        if not self.llm_manager and self.service_container:
            try:
                self.llm_manager = await self.service_container.get_service(
                    "llm_manager"
                )
            except Exception as e:  # pragma: no cover - service retrieval issue
                integration_service_logger.warning(
                    "LLMManager unavailable for summarization", exception=e
                )
        llm_manager = self.llm_manager
        if not llm_manager:
            return text[:max_tokens]
        prompt = (
            f"Summarize the following text in {max_tokens} tokens or fewer:\n{text}"
        )
        try:
            response = await llm_manager.complete(prompt, max_tokens=max_tokens)
            return response.content
        except Exception as e:  # pragma: no cover - LLM failure
            integration_service_logger.error("LLM summarization failed", exception=e)
            return text[:max_tokens]

    @detailed_log_function(LogCategory.API)
    async def get_system_status_summary(self) -> Dict[str, Any]:  # Renamed
        """Aggregates status from various core services."""
        integration_service_logger.info("Fetching system status summary.")

        if not self.service_container or not hasattr(
            self.service_container, "get_system_health_summary"
        ):
            integration_service_logger.error(
                "Service container not available or missing health summary method."
            )
            return {
                "overall_status": "unavailable",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        try:
            status_summary = await self.service_container.get_system_health_summary()
        except Exception as e:  # pragma: no cover - container failure
            integration_service_logger.error(
                "Failed to obtain system health summary.", exception=e
            )
            return {
                "overall_status": "error",
                "details": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Gather additional metrics from workflow orchestrator or managers
        try:
            orchestrator = await self.service_container.get_service(
                "ultimate_orchestrator"
            )
        except Exception:
            orchestrator = None

        if not orchestrator:
            try:
                orchestrator = await self.service_container.get_service(
                    "realtime_analysis_workflow"
                )
            except Exception:
                orchestrator = None

        if orchestrator and hasattr(orchestrator, "get_system_stats"):
            try:
                status_summary["workflow_metrics"] = await orchestrator.get_system_stats()
            except Exception as e:
                integration_service_logger.warning(
                    "Workflow metrics unavailable.", exception=e
                )

        # Add metrics from realtime graph manager if available
        try:
            rt_graph_manager = await self.service_container.get_service(
                "realtime_graph_manager"
            )
            if rt_graph_manager and hasattr(rt_graph_manager, "get_realtime_stats"):
                status_summary["realtime_graph_stats"] = await rt_graph_manager.get_realtime_stats()
        except Exception as e:
            integration_service_logger.warning(
                "Real-time graph stats unavailable.", exception=e
            )

        # Include reviewable memory statistics for pending reviews
        try:
            review_memory = await self.service_container.get_service(
                "reviewable_memory"
            )
            if review_memory and hasattr(review_memory, "get_review_stats_async"):
                status_summary["review_memory_stats"] = await review_memory.get_review_stats_async()
        except Exception as e:
            integration_service_logger.warning(
                "Review memory stats unavailable.", exception=e
            )

        integration_service_logger.info(
            "System status summary retrieved.", parameters=status_summary
        )
        return status_summary

    # Add other methods that FastAPI endpoints will call, e.g.:
    # async def search_knowledge_graph(...)
    # async def submit_review_decision_service(...)

    @detailed_log_function(LogCategory.API)
    async def run_analogical_reasoning(self, base_case: str, target_case: str) -> Dict[str, Any]:
        """Run analogical reasoning between two cases using the LegalReasoningEngine."""
        if not self.legal_reasoning_engine:
            try:
                self.legal_reasoning_engine = await self.service_container.get_service("legal_reasoning_engine")
            except Exception as e:
                integration_service_logger.error(
                    "LegalReasoningEngine retrieval failed.", exception=e
                )
                raise ServiceLayerError("LegalReasoningEngine not available", cause=e)

        engine = self.legal_reasoning_engine
        if not engine:
            raise ServiceLayerError("LegalReasoningEngine not available")

        try:
            return await engine.run_analogical_reasoning(base_case, target_case)
        except Exception as e:  # pragma: no cover - engine failure
            integration_service_logger.error("Analogical reasoning failed", exception=e)
            raise ServiceLayerError("Analogical reasoning failed", cause=e)

    @detailed_log_function(LogCategory.API)
    async def interpret_statute(self, statute_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Interpret statutory language in context."""
        if not self.legal_reasoning_engine:
            try:
                self.legal_reasoning_engine = await self.service_container.get_service("legal_reasoning_engine")
            except Exception as e:
                integration_service_logger.error(
                    "LegalReasoningEngine retrieval failed.", exception=e
                )
                raise ServiceLayerError("LegalReasoningEngine not available", cause=e)

        engine = self.legal_reasoning_engine
        if not engine:
            raise ServiceLayerError("LegalReasoningEngine not available")

        try:
            return await engine.interpret_statute(statute_text, context)
        except Exception as e:  # pragma: no cover - engine failure
            integration_service_logger.error("Statute interpretation failed", exception=e)
            raise ServiceLayerError("Statute interpretation failed", cause=e)

    @detailed_log_function(LogCategory.API)
    async def perform_constitutional_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze text for constitutional issues."""
        if not self.legal_reasoning_engine:
            try:
                self.legal_reasoning_engine = await self.service_container.get_service("legal_reasoning_engine")
            except Exception as e:
                integration_service_logger.error(
                    "LegalReasoningEngine retrieval failed.", exception=e
                )
                raise ServiceLayerError("LegalReasoningEngine not available", cause=e)

        engine = self.legal_reasoning_engine
        if not engine:
            raise ServiceLayerError("LegalReasoningEngine not available")

        try:
            return await engine.perform_constitutional_analysis(text)
        except Exception as e:  # pragma: no cover - engine failure
            integration_service_logger.error(
                "Constitutional analysis failed", exception=e
            )
            raise ServiceLayerError("Constitutional analysis failed", cause=e)

    @detailed_log_function(LogCategory.API)
    async def predict_case_outcome(self, case_facts: str) -> Dict[str, Any]:
        """Predict likely case outcome based on provided facts."""
        if not self.legal_reasoning_engine:
            try:
                self.legal_reasoning_engine = await self.service_container.get_service("legal_reasoning_engine")
            except Exception as e:
                integration_service_logger.error(
                    "LegalReasoningEngine retrieval failed.", exception=e
                )
                raise ServiceLayerError("LegalReasoningEngine not available", cause=e)

        engine = self.legal_reasoning_engine
        if not engine:
            raise ServiceLayerError("LegalReasoningEngine not available")

        try:
            return await engine.predict_case_outcome(case_facts)
        except Exception as e:  # pragma: no cover - engine failure
            integration_service_logger.error("Case outcome prediction failed", exception=e)
            raise ServiceLayerError("Case outcome prediction failed", cause=e)

    async def initialize_service(self):  # For service container
        integration_service_logger.info(
            "LegalAIIntegrationService (async) initialize called."
        )
        # Attempt to retrieve core services now that the container is initializing
        if not self.security_manager:
            try:
                self.security_manager = await self.service_container.get_service(
                    "security_manager"
                )
            except Exception as e:
                integration_service_logger.debug(
                    "SecurityManager not available during initialization.",
                    exception=e,
                )
        if not self.llm_manager:
            try:
                self.llm_manager = await self.service_container.get_service(
                    "llm_manager"
                )
            except Exception as e:
                integration_service_logger.debug(
                    "LLMManager not available during initialization.",
                    exception=e,
                )
        return self

    async def get_service_status(self) -> Dict[str, Any]:  # For service container
        # A simple health check for the integration service itself
        return {
            "status": "healthy" if self.service_container else "degraded_no_container",
            "service_name": "LegalAIIntegrationService",
            "dependencies_status": {
                "service_container": (
                    "available" if self.service_container else "unavailable"
                ),
                "security_manager": (
                    "available" if self.security_manager else "unavailable"
                ),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# Factory for service container
def create_integration_service(
    service_container: ServiceContainer,
) -> LegalAIIntegrationService:
    return LegalAIIntegrationService(service_container)
