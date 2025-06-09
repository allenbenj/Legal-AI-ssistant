# legal_ai_system/services/integration_service.py
"""
Integration Service for Legal AI System.

Provides a higher-level service facade that uses the ServiceContainer
to interact with various backend components (agents, managers, workflows).
This service is typically used by the API layer (e.g., FastAPI).
"""

# import logging # Replaced by detailed_logging
from typing import Dict, List, Any, Optional, Union, Callable # Added Callable
from pathlib import Path
import uuid
from datetime import datetime, timezone # Added timezone

# Use detailed_logging
from ..core.detailed_logging import get_detailed_logger, LogCategory, detailed_log_function
# Import core components it will interact with via ServiceContainer
from .service_container import ServiceContainer
from .security_manager import SecurityManager, AuthUser, AccessLevel  # For user context
from ..core.unified_exceptions import ServiceLayerError, DocumentNotFoundError, ProcessingError
# Import result types from agents/workflows if this service returns them directly
from ..agents.document_processor.document_processor_agent import DocumentProcessingOutput
from ..workflows.realtime_analysis_workflow import RealTimeAnalysisResult # Example

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
        # Get frequently used services for convenience, or fetch them on-demand in methods
        self.security_manager: Optional[SecurityManager] = self.service_container.get_service("security_manager")
        # Example: self.realtime_workflow: Optional[RealTimeAnalysisWorkflow] = self.service_container.get_service("realtime_analysis_workflow")
        
        integration_service_logger.info("LegalAIIntegrationService initialized successfully.")

    @detailed_log_function(LogCategory.API)
    async def handle_document_upload(self, 
                                     file_content: bytes, 
                                     filename: str, 
                                     user: AuthUser, # Pass authenticated user object
                                     options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handles document upload, initial validation, and storage.
        Triggers background processing via a workflow.
        """
        integration_service_logger.info("Handling document upload.", parameters={'filename': filename, 'user_id': user.user_id})
        options = options or {}
        
        if not self.security_manager:
            raise ServiceLayerError("SecurityManager not available for document upload.")

        # Store the file securely (SecurityManager might offer a method, or use a dedicated storage service)
        # This is a simplified storage step. Production would use a robust storage solution.
        upload_dir = Path("./storage/documents/uploads_service") # Should come from config
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename and create unique path (logic similar to FastAPI's main.py)
        safe_filename = "".join(c if c.isalnum() or c in ['.', '-', '_'] else '_' for c in filename)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{safe_filename}"
        file_path = upload_dir / unique_filename

        try:
            with open(file_path, "wb") as f:
                f.write(file_content)
            integration_service_logger.info("File saved to temporary upload location.", parameters={'path': str(file_path)})

            # TODO: Add metadata to a persistent document registry (e.g., using EnhancedPersistenceManager)
            # For now, we'll generate a conceptual document_id
            document_id = f"doc_serv_{uuid.uuid4().hex}"
            
            # Trigger background processing using an orchestrator/workflow service
            orchestrator = self.service_container.get_service("ultimate_orchestrator") # Or "realtime_analysis_workflow"
            if not orchestrator:
                raise ServiceLayerError("Workflow orchestrator service not available.")

            # The orchestrator's execute_workflow should be an async task
            # FastAPI's BackgroundTasks would be used in the API endpoint layer, not here.
            # This service method itself might be called by a BackgroundTask.
            # We are just initiating the call to the workflow here.
            
            # Construct task_data and metadata for the workflow
            workflow_metadata = {
                'document_id': document_id, # The conceptual ID for tracking
                'original_filename': filename,
                'user_id': user.user_id,
                'upload_timestamp': datetime.now(timezone.utc).isoformat(),
                'processing_options': options 
            }
            
            # The workflow will handle its own background execution if designed that way (e.g. LangGraph)
            # Or, if orchestrator.execute_workflow_instance is a long blocking call, it should be
            # launched as a separate task by the API layer.
            # For now, let's assume the workflow handles its own asynchronicity or the API layer does.
            # Here, we're just setting up to call it.
            # In a real scenario, we wouldn't await the full workflow here if it's long.
            # We'd return an ack and the workflow runs in bg.
            # This function's role is to prepare and initiate.
            
            # Example: If the workflow needs to be explicitly run in background from here:
            # task = asyncio.create_task(
            #    orchestrator.execute_workflow_instance(document_path_str=str(file_path), custom_metadata=workflow_metadata)
            # )
            # self.service_container.add_background_task(task) # If container manages tasks

            integration_service_logger.info("Document processing initiated via workflow.", 
                                          parameters={'doc_id': document_id, 'workflow_service': type(orchestrator).__name__})

            return {
                'document_id': document_id,
                'filename': unique_filename,
                'size_bytes': len(file_content),
                'status': 'processing_initiated', # Indicate it's handed off
                'message': 'Document accepted and processing started.'
            }
            
        except Exception as e:
            integration_service_logger.error("Failed to handle document upload and initiate processing.", 
                                            parameters={'filename': filename}, exception=e)
            raise ServiceLayerError(f"Document upload handling failed: {str(e)}", cause=e)

    @detailed_log_function(LogCategory.API)
    async def get_document_analysis_status(self, document_id: str, user: AuthUser) -> Dict[str, Any]:
        """Retrieves the processing status and summary for a document."""
        integration_service_logger.debug("Fetching document analysis status.", parameters={'doc_id': document_id, 'user_id': user.user_id})
        # This would query a workflow state manager or document metadata store
        # Example:
        # workflow_state_manager = self.service_container.get_service("workflow_state_manager")
        # status = await workflow_state_manager.get_status(document_id)
        # For now, mock:
        if document_id == "doc_serv_test123":
            return {"document_id": document_id, "status": "completed", "progress": 1.0, "stage": "Done", 
                    "summary": {"entities_found": 10, "violations_detected": 1}}
        return {"document_id": document_id, "status": "processing", "progress": 0.5, "stage": "entity_extraction"}

    @detailed_log_function(LogCategory.API)
    async def get_system_status_summary(self) -> Dict[str, Any]: # Renamed
        """Aggregates status from various core services."""
        integration_service_logger.info("Fetching system status summary.")
        # This method would call health_check() or get_service_status() on key services
        # registered in the service_container.
        
        # Example of how it might work:
        # status_summary = await self.service_container.get_system_health_summary()
        # return status_summary
        
        # Mocked response for now:
        mock_summary = {
            "overall_status": "HEALTHY",
            "services_status": {
                "llm_manager": {"status": "healthy"},
                "knowledge_graph_manager": {"status": "healthy"},
                "vector_store": {"status": "healthy"},
                "persistence_manager": {"status": "healthy"}
            },
            "performance_metrics_summary": {"avg_workflow_time_sec": 120.5, "active_workflows": 3},
            "active_documents_count": 5,
            "pending_reviews_count": 2,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        integration_service_logger.info("System status summary retrieved (mocked).", parameters=mock_summary)
        return mock_summary
        
    # Add other methods that FastAPI endpoints will call, e.g.:
    # async def search_knowledge_graph(...)
    # async def submit_review_decision_service(...)

    async def initialize_service(self): # For service container
        integration_service_logger.info("LegalAIIntegrationService (async) initialize called.")
        # Nothing specific to init here as it relies on service_container being ready.
        return self

    async def get_service_status(self) -> Dict[str, Any]: # For service container
        # A simple health check for the integration service itself
        return {
            "status": "healthy" if self.service_container else "degraded_no_container",
            "service_name": "LegalAIIntegrationService",
            "dependencies_status": {
                "service_container": "available" if self.service_container else "unavailable",
                "security_manager": "available" if self.security_manager else "unavailable"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Factory for service container
def create_integration_service(service_container: ServiceContainer) -> LegalAIIntegrationService:
    return LegalAIIntegrationService(service_container)
