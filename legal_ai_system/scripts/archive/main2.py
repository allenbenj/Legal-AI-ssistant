# legal_ai_system/main.py (Further Refinements)
"""FastAPI Backend for Legal AI System."""

# ... (imports from previous main.py refactor, ensure they are correct for new structure)
import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone # Added timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union 
import uuid 
import os

import uvicorn
from fastapi import (
    FastAPI, Depends, HTTPException, status, UploadFile, File, Form, 
    WebSocket, WebSocketDisconnect, BackgroundTasks
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse

import strawberry # type: ignore
from strawberry.fastapi import GraphQLRouter # type: ignore
from strawberry.types import Info # type: ignore
from pydantic import BaseModel, Field as PydanticField
from jose import JWTError, jwt # type: ignore

from core.detailed_logging import get_detailed_logger, LogCategory
from legal_ai_system.core.constants import Constants
from core.service_container import ServiceContainer, create_service_container
from core.security_manager import SecurityManager, AccessLevel, AuthUser # Using aliased AuthUser
from services.integration_service import LegalAIIntegrationService, create_integration_service # Assuming factory
from workflows.realtime_analysis_workflow import RealTimeAnalysisWorkflow, RealTimeAnalysisResult
# Import specific request/response models if they are defined elsewhere
# from core.models import DocumentProcessingOptions, ProcessingStatusResponse, ...

main_api_logger = get_detailed_logger("FastAPI_App", LogCategory.API)

# --- Global Instances (Initialized in Lifespan) ---
service_container_instance: Optional[ServiceContainer] = None
security_manager_instance: Optional[SecurityManager] = None
websocket_manager_instance: Optional['WebSocketManager'] = None
integration_service_instance: Optional[LegalAIIntegrationService] = None

# --- Document Store (Simple In-Memory for this example) ---
# In a real system, this would be handled by EnhancedPersistenceManager or a DocumentRegistryService
# Format: { "doc_id_xyz": {"file_path": "/path/to/file.pdf", "original_filename": "file.pdf", "status": "uploaded", ...} }
DOCUMENT_METADATA_STORE: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service_container_instance, security_manager_instance, websocket_manager_instance, integration_service_instance
    main_api_logger.info("🚀 FastAPI Lifespan: Application startup...")
    try:
        # 1. Create Service Container (which loads ConfigurationManager)
        # Pass app_settings if LegalAISettings is instantiated here, else CM loads defaults
        from config.settings import settings as global_app_settings
        service_container_instance = await create_service_container(app_settings=global_app_settings)
        main_api_logger.info("ServiceContainer initialized.")

        # 2. Get/Initialize critical services from container
        security_manager_instance = await service_container_instance.get_service("security_manager")
        main_api_logger.info("SecurityManager obtained from container.")
        
        # Create and register IntegrationService
        # integration_service_instance = LegalAIIntegrationService(service_container_instance)
        # await service_container_instance.register_service("integration_service", instance=integration_service_instance)
        # await integration_service_instance.initialize_service() # If it has one
        # OR use a factory if preferred by container design
        integration_service_instance = await service_container_instance.get_service("integration_service")

        main_api_logger.info("IntegrationService obtained/initialized.")

        websocket_manager_instance = WebSocketManager() # Assuming it's simple enough not to be a managed service
        main_api_logger.info("WebSocketManager initialized.")
        
        main_api_logger.info("✅ Legal AI System API ready via lifespan.")
    except Exception as e:
        main_api_logger.critical("FATAL: Error during API startup lifespan.", exception=e)
        # Optionally, re-raise to prevent FastAPI from starting if critical components fail
        raise SystemExit(f"API startup failed: {e}") from e
    
    yield # API is running
    
    main_api_logger.info("🛑 FastAPI Lifespan: Application shutdown...")
    if service_container_instance:
        await service_container_instance.shutdown_all_services()
        main_api_logger.info("All services in container shut down.")
    main_api_logger.info("Legal AI System API shutdown complete.")

app = FastAPI(
    title="Legal AI System API (Refactored)",
    description="Refactored API for Legal AI document processing and analysis.",
    version=Constants.Version.APP_VERSION if hasattr(Constants, "Version") else "2.1.0",
    lifespan=lifespan
)
# ... (CORS, Security Scheme, JWT constants as before) ...
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
security_scheme = HTTPBearer()
JWT_SECRET_KEY = os.getenv("LEGAL_AI_JWT_SECRET_KEY", "a_very_secret_key_for_jwt_replace_me_in_production_XYZ")
JWT_ALGORITHM = "HS256"


# --- Pydantic Models (Request/Response - from previous refactor) ---
class TokenResponse(BaseModel): access_token: str; token_type: str; user: Dict[str, Any]
class LoginRequest(BaseModel): username: str; password: str
class DocumentUploadResponse(BaseModel): document_id: str; filename: str; size_bytes: int; status: str; message: Optional[str] = None
class ProcessingRequest(BaseModel):
    processing_options: Dict[str, Any] = PydanticField(default_factory=dict, description="Agent/workflow specific options")
    priority: Optional[str] = PydanticField("normal", description="Task priority: low, normal, high, urgent")
class DocumentStatusResponse(BaseModel): document_id: str; status: str; progress: float; stage: Optional[str] = None; result_summary: Optional[Dict[str, Any]] = None
class ReviewDecisionRequest(BaseModel): item_id: str; decision: str; modified_data: Optional[Dict[str, Any]] = None; reviewer_notes: Optional[str] = None; reviewer_id: str # Added reviewer_id
class SystemHealthResponse(BaseModel): overall_status: str; services_status: Dict[str, Any]; performance_metrics_summary: Dict[str, Any]; active_documents_count: int; pending_reviews_count: int; timestamp: str

# --- Auth Endpoints (Integrated with SecurityManager) ---
@app.post("/api/v1/auth/token", response_model=TokenResponse)
async def login_for_access_token_api(form_data: LoginRequest): # Renamed
    main_api_logger.info("Login attempt via API.", parameters={'username': form_data.username})
    if not security_manager_instance:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication service not available.")
    
    session_token_internal = security_manager_instance.auth_manager.authenticate(
        form_data.username, form_data.password, ip_address="api_request" # TODO: Get actual IP
    )
    if not session_token_internal:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    
    user_obj = next((u for u in security_manager_instance.auth_manager.users.values() if u.username == form_data.username), None)
    if not user_obj: # Should not happen if authenticate passed
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User data not found after auth.")

    user_info_for_token = {"user_id": user_obj.user_id, "username": user_obj.username, "roles": [user_obj.access_level.value]}
    jwt_token = create_access_token(data=user_info_for_token)
    
    user_info_for_response = {"user_id": user_obj.user_id, "username": user_obj.username, "email": user_obj.email, "access_level": user_obj.access_level.value}
    return TokenResponse(access_token=jwt_token, token_type="bearer", user=user_info_for_response)

async def get_current_active_user_from_token(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> AuthUser:
    """Validates JWT and returns user. Actual implementation."""
    if not security_manager_instance:
        main_api_logger.warning("Bypassing token validation: SecurityManager not available.")
        return AuthUser(user_id="mock_user_sec_off", username="test_user_sec_off", email="test@example.com", access_level=AccessLevel.ADMIN, last_login=datetime.now(timezone.utc))

    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: Optional[str] = payload.get("user_id")
        username: Optional[str] = payload.get("sub") # Standard subject claim
        if user_id is None and username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token: missing user identifier.")
        
        # Fetch user from SecurityManager's user store
        user = security_manager_instance.auth_manager.users.get(user_id) if user_id else \
               next((u for u in security_manager_instance.auth_manager.users.values() if u.username == username), None)

        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found for token.")
        if not user.is_active:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user.")
        return user
    except JWTError as e:
        main_api_logger.warning("JWTError during token validation.", exception=e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

def require_permission_api(required_level: AccessLevel): # Renamed
    """Dependency factory for permission checking using actual SecurityManager."""
    async def permission_checker_impl(current_user: AuthUser = Depends(get_current_active_user_from_token)) -> AuthUser: # Renamed
        if not security_manager_instance:
             main_api_logger.warning("Bypassing permission check (SecurityManager unavailable). Granting access.")
             return current_user # Bypass if security manager isn't up
        
        if not security_manager_instance.auth_manager.check_permission(current_user, required_level):
            main_api_logger.warning("Permission denied for user.", 
                                   parameters={'user_id': current_user.user_id, 'required': required_level.value, 'actual': current_user.access_level.value})
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Operation not permitted for your access level.")
        return current_user
    return permission_checker_impl


@app.get("/api/v1/auth/me", response_model=Dict[str, Any])
async def read_users_me_api(current_user: AuthUser = Depends(get_current_active_user_from_token)): # Renamed, uses new auth
    main_api_logger.info("API: Fetching current user info.", parameters={'user_id': current_user.user_id})
    return {
        "user_id": current_user.user_id, "username": current_user.username, "email": current_user.email,
        "access_level": current_user.access_level.value if isinstance(current_user.access_level, Enum) else current_user.access_level,
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None,
        "is_active": current_user.is_active
    }

# --- Document Endpoints (Using IntegrationService) ---
@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document_api(
    file: UploadFile = File(...),
    current_user: AuthUser = Depends(require_permission_api(AccessLevel.WRITE))
):
    main_api_logger.info("API: Document upload request.", parameters={'filename': file.filename, 'user_id': current_user.user_id})
    if not integration_service_instance:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Document handling service not available.")
    
    try:
        content = await file.read()
        # Store file temporarily, IntegrationService or a StorageService should handle permanent storage
        temp_dir = Path("./storage/temp_uploads") # Should come from config
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename
        safe_filename = "".join(c if c.isalnum() or c in ['.', '-', '_'] else '_' for c in file.filename or "unknown_file")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        temp_file_path = temp_dir / f"{timestamp}_{uuid.uuid4().hex[:8]}_{safe_filename}"

        with open(temp_file_path, "wb") as f_out:
            f_out.write(content)

        # The IntegrationService's handle_document_upload should take the file path
        # and then decide where to store it permanently, add to DB etc.
        # For now, we pass content and original filename.
        # A better approach: IS takes path, reads it, and manages it.
        
        # Simplified: IS takes content. For a real system, IS would take path or work with a StorageService.
        upload_response_dict = await integration_service_instance.handle_document_upload(
            file_content=content, # Pass content bytes
            filename=file.filename or "uploaded_file", # Original filename
            user=current_user, # Pass AuthUser object
            options={} # Pass any initial options if applicable
        )
        # Store mapping of conceptual document_id to actual file_path
        DOCUMENT_METADATA_STORE[upload_response_dict['document_id']] = {
            "file_path": str(temp_file_path), # Store the path where it's actually saved
            "original_filename": file.filename or "uploaded_file",
            "status": "uploaded",
            "user_id": current_user.user_id,
            "size_bytes": len(content)
        }
        return DocumentUploadResponse(**upload_response_dict)

    except ServiceLayerError as sle: # Catch specific errors from service
        main_api_logger.error("ServiceLayerError during document upload.", exception=sle)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(sle))
    except Exception as e:
        main_api_logger.error("Unexpected error during document upload.", parameters={'filename': file.filename}, exception=e)
        # Clean up temp file if created
        if 'temp_file_path' in locals() and temp_file_path.exists(): temp_file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Upload failed: {str(e)}")


@app.post("/api/v1/documents/{document_id}/process", status_code=status.HTTP_202_ACCEPTED)
async def process_document_api(
    document_id: str,
    processing_request: ProcessingRequest, # FastAPI will parse JSON body into this model
    background_tasks: BackgroundTasks,
    current_user: AuthUser = Depends(require_permission_api(AccessLevel.WRITE))
):
    main_api_logger.info("API: Request to process document.", parameters={'doc_id': document_id, 'user_id': current_user.user_id})
    if not integration_service_instance:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Processing service not available.")

    doc_meta = DOCUMENT_METADATA_STORE.get(document_id)
    if not doc_meta or not Path(doc_meta["file_path"]).exists():
        main_api_logger.error(f"Document file for ID '{document_id}' not found in metadata store or filesystem.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {document_id} not found or file missing.")
    
    document_file_path_to_process = doc_meta["file_path"]

    # Update status in our mock store
    DOCUMENT_METADATA_STORE[document_id]["status"] = "queued_for_processing"
    DOCUMENT_METADATA_STORE[document_id]["processing_options"] = processing_request.model_dump()

    # The background task should be defined to call the IntegrationService method
    # which then calls the workflow.
    async def _background_processing_wrapper(doc_id: str, file_path_str: str, req: ProcessingRequest, user: AuthUser):
        if integration_service_instance: # Should always be true if we reach here
            try:
                # This method on IntegrationService would invoke the UltimateOrchestrator
                # await integration_service_instance.initiate_and_run_workflow(
                #    document_id=doc_id, 
                #    document_actual_path=file_path_str, 
                #    processing_options=req.processing_options, # Pass the dict
                #    priority=req.priority, 
                #    requesting_user=user
                # )
                # For now, directly calling the global background task for simplicity of this step
                # This assumes process_document_background_task is defined and accessible.
                # Ideally, this logic is inside IntegrationService.
                await process_document_background_task_api(doc_id, file_path_str, req, user.user_id)
            except Exception as e:
                main_api_logger.error("Error in background processing task from API layer.", 
                                     parameters={'doc_id': doc_id}, exception=e)
                DOCUMENT_METADATA_STORE[doc_id]["status"] = "failed"
                DOCUMENT_METADATA_STORE[doc_id]["error"] = str(e)
                if websocket_manager_instance:
                    await websocket_manager_instance.broadcast_to_topic({
                        "type": "processing_error", "document_id": doc_id, "error": str(e)
                    }, f"doc_updates_{doc_id}")


    background_tasks.add_task(
        _background_processing_wrapper,
        document_id,
        document_file_path_to_process,
        processing_request, # Pass the Pydantic model
        current_user
    )
    main_api_logger.info("Document processing task added to background via API.", parameters={'document_id': document_id})
    return {"message": "Document processing initiated.", "document_id": document_id}


@app.get("/api/v1/documents/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status_api(
    document_id: str,
    current_user: AuthUser = Depends(require_permission_api(AccessLevel.READ))
):
    main_api_logger.debug("API: Request for document status.", parameters={'doc_id': document_id, 'user_id': current_user.user_id})
    if not integration_service_instance:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Status service not available.")
    
    # status_info = await integration_service_instance.get_document_analysis_status(document_id, current_user)
    # Mocking based on global_processing_states or DOCUMENT_METADATA_STORE
    doc_meta = DOCUMENT_METADATA_STORE.get(document_id)
    state_info = global_processing_states.get(document_id) if 'global_processing_states' in globals() else None

    if state_info: # If actively being processed by background task
        return DocumentStatusResponse(
            document_id=document_id, status=state_info.get("status", "unknown"),
            progress=state_info.get("progress", 0.0) * 100, # Convert 0-1 to 0-100 for display
            stage=state_info.get("stage"),
            result_summary=state_info.get("result_summary")
        )
    elif doc_meta: # If in metadata store but not actively processing
        return DocumentStatusResponse(
            document_id=document_id, status=doc_meta.get("status", "unknown"),
            progress=100.0 if doc_meta.get("status") == "completed" else (0.0 if doc_meta.get("status") == "uploaded" else 5.0), # Basic progress
            stage=doc_meta.get("status")
        )
    
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Status for document {document_id} not found.")


@app.get("/api/v1/system/health", response_model=SystemHealthResponse)
async def get_system_health_api(current_user: AuthUser = Depends(require_permission_api(AccessLevel.ADMIN))): # Admin only
    main_api_logger.info("API: System health check requested.", parameters={'user_id': current_user.user_id})
    if not integration_service_instance:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="System status service not available.")
    try:
        health_summary = await integration_service_instance.get_system_status_summary()
        return SystemHealthResponse(**health_summary)
    except Exception as e:
        main_api_logger.error("API: Failed to get system health.", exception=e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Health check failed: {str(e)}")

@app.post("/api/v1/calibration/review", status_code=status.HTTP_200_OK)
async def submit_review_decision_api( # Renamed
    review_request: ReviewDecisionRequest,
    current_user: AuthUser = Depends(require_permission_api(AccessLevel.WRITE))
):
    main_api_logger.info("API: Review decision submitted.", parameters={'item_id': review_request.item_id, 'user_id': current_user.user_id})
    if not integration_service_instance:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Review service not available.")
    
    # Assuming IntegrationService has a method to handle this
    # success = await integration_service_instance.submit_review_for_item(review_request, current_user)
    # Mocking for now
    mock_review_service = service_container_instance.get_service("reviewable_memory") if service_container_instance else None
    if mock_review_service and hasattr(mock_review_service, "submit_review_decision_async"):
        from memory.reviewable_memory import ReviewDecision as ReviewDecisionDataclass, ReviewStatus
        # Map string decision to Enum
        try:
            decision_enum = ReviewStatus[review_request.decision.upper()]
        except KeyError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid review decision string: {review_request.decision}")

        decision_dc = ReviewDecisionDataclass(
            item_id=review_request.item_id,
            decision=decision_enum,
            modified_content=review_request.modified_data,
            reviewer_notes=review_request.reviewer_notes or "",
            reviewer_id=current_user.user_id
        )
        success = await mock_review_service.submit_review_decision_async(decision_dc)
    else:
        success = True # Mock success if service not fully available

    if success:
        if websocket_manager_instance:
            await websocket_manager_instance.broadcast_to_topic({
                "type": "review_processed", "item_id": review_request.item_id, "decision": review_request.decision,
                "user": current_user.username, "timestamp": datetime.now(timezone.utc).isoformat()
            }, "calibration_updates")
        return {"status": "review_processed", "item_id": review_request.item_id}
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to process review decision via service.")


# --- WebSocket Endpoint (from previous refactor, ensure websocket_manager_instance is used) ---
@app.websocket("/ws/{client_id}") # Changed user_id to client_id for clarity if user isn't auth'd yet for WS
async def websocket_endpoint_api(websocket: WebSocket, client_id: str): # Renamed
    if not websocket_manager_instance:
        main_api_logger.error("WebSocket connection attempt failed: WebSocketManager not initialized.")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR); return

    # TODO: Authenticate WebSocket connection here if required, e.g., via token in query param or first message.
    # For now, using client_id from path as user_id.
    user_id_for_ws = client_id 
    await websocket_manager_instance.connect(websocket, user_id_for_ws)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data) # Basic JSON parsing
            main_api_logger.debug("WebSocket message received.", parameters={'client_id': client_id, 'msg_type': message.get("type")})
            
            msg_type = message.get("type")
            if msg_type == "subscribe":
                topic = message.get("topic")
                if topic and isinstance(topic, str): await websocket_manager_instance.subscribe_to_topic(user_id_for_ws, topic)
            # ... (other handlers as before) ...
    except WebSocketDisconnect:
        main_api_logger.info(f"WebSocket client disconnected.", parameters={'client_id': client_id})
    except Exception as e:
        main_api_logger.error(f"WebSocket error for client.", parameters={'client_id': client_id}, exception=e)
    finally:
        if websocket_manager_instance: websocket_manager_instance.disconnect(user_id_for_ws)


# --- Background Task (adapted from original, now called by API endpoint) ---
# This should ideally be part of IntegrationService or a WorkflowService
async def process_document_background_task_api( # Renamed
    document_id: str, 
    document_file_path_str: str, # Actual path
    processing_request: ProcessingRequest, # Pydantic model
    requesting_user_id: str
):
    main_api_logger.info("Background task API: Starting processing.", 
                       parameters={'doc_id': document_id, 'file_path': document_file_path_str, 'user_id': requesting_user_id})
    
    global_processing_states[document_id] = {"status": "processing_setup", "progress": 0.05, "stage": "Workflow Initialization"}

    if not service_container_instance:
        main_api_logger.critical("Background task API: ServiceContainer not available.")
        # Update global state and notify WebSocket
        global_processing_states[document_id].update({"status":"failed", "error": "System service container unavailable."})
        if websocket_manager_instance:
             await websocket_manager_instance.broadcast_to_topic({
                "type": "processing_error", "document_id": document_id, "error": "Service container unavailable."
            }, f"doc_updates_{document_id}")
        return

    try:
        # Get the orchestrator (e.g., UltimateWorkflowOrchestrator or RealTimeAnalysisWorkflow)
        orchestrator = await service_container_instance.get_service("ultimate_orchestrator")
        if not orchestrator:
            raise ServiceLayerError("UltimateOrchestrator service not found.")

        # Prepare metadata for the workflow
        workflow_custom_metadata = {
            'document_id': document_id, # Conceptual ID
            'original_filename': Path(document_file_path_str).name, # From actual path
            'user_id': requesting_user_id,
            'upload_timestamp': DOCUMENT_METADATA_STORE.get(document_id, {}).get("upload_timestamp", datetime.now(timezone.utc).isoformat()),
            'api_processing_options': processing_request.processing_options, # Pass options from API request
            'api_priority': processing_request.priority
        }
        
        # TODO: The orchestrator should ideally have a progress callback mechanism that this task can subscribe to
        # for more granular WebSocket updates. For now, we update WebSocket before/after.
        if websocket_manager_instance: # Initial progress update
            await websocket_manager_instance.broadcast_to_topic({
                "type": "processing_progress", "document_id": document_id, "progress": 10, "stage": "Workflow Execution Starting"
            }, f"doc_updates_{document_id}")


        # Execute the workflow
        # orchestrator.execute_workflow_instance is async
        workflow_result_state: OrchestratorWorkflowState = await orchestrator.execute_workflow_instance(
            document_path_str=document_file_path_str, # Pass actual file path
            custom_metadata=workflow_custom_metadata
        )
        
        # Update global state based on workflow result
        final_status = workflow_result_state.current_status.value
        final_progress = 1.0 if final_status == WorkflowStatus.COMPLETED.value else (global_processing_states[document_id].get("progress", 0.9) if final_status == WorkflowStatus.FAILED.value else 0.98)
        
        global_processing_states[document_id].update({
            "status": final_status, 
            "progress": final_progress, 
            "stage": "Workflow Finished",
            "result_summary": {
                "total_time_sec": workflow_result_state.total_workflow_processing_time_sec,
                "errors": workflow_result_state.error_messages_list,
                # Add more summary from workflow_result_state.payload if needed
            }
        })

        if websocket_manager_instance:
            event_type = "processing_complete" if final_status == WorkflowStatus.COMPLETED.value else "processing_error"
            await websocket_manager_instance.broadcast_to_topic({
                "type": event_type, "document_id": document_id,
                "result_summary": global_processing_states[document_id]["result_summary"],
                # "full_result": workflow_result_state.to_dict() # Optionally send full result
            }, f"doc_updates_{document_id}") # Topic per document ID

        main_api_logger.info("Background task API: Processing finished.", 
                           parameters={'doc_id': document_id, 'status': final_status, 'total_time': workflow_result_state.total_workflow_processing_time_sec})

    except Exception as e:
        main_api_logger.error(f"Background task API: Processing critically failed.", 
                             parameters={'doc_id': document_id}, exception=e)
        global_processing_states[document_id].update({"status": "failed", "error": f"Critical task error: {str(e)}"})
        if websocket_manager_instance:
            await websocket_manager_instance.broadcast_to_topic({
                "type": "processing_error", "document_id": document_id, "error": f"Critical task error: {str(e)}"
            }, f"doc_updates_{document_id}")


# --- Static Files (for frontend) ---
# Serve frontend if built and path configured
# This should be configurable via ConfigurationManager
# frontend_path_from_config = service_container_instance.get_service("configuration_manager").get("frontend_dist_path", None)
# if frontend_path_from_config and Path(frontend_path_from_config).exists():
#    app.mount("/", StaticFiles(directory=str(frontend_path_from_config), html=True), name="static_frontend")
#    main_api_logger.info(f"Serving static frontend from: {frontend_path_from_config}")
# else:
#    main_api_logger.info("Static frontend serving not configured or path not found.")


if __name__ == "__main__":
    # This setup is for running this FastAPI app directly.
    # The lifespan function handles initialization of services.
    if not main_api_logger.logger.hasHandlers(): # Basic logging if not configured by detailed_logging yet
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main_api_logger.info("Starting Legal AI System FastAPI server via __main__.")
    uvicorn.run(
        "main:app", 
        host=os.getenv("LEGAL_AI_API_HOST", "0.0.0.0"), # Get from env or default
        port=int(os.getenv("LEGAL_AI_API_PORT", "8000")),
        reload=True, # Enable for development
        log_level=os.getenv("LEGAL_AI_API_LOG_LEVEL", "info").lower()
    )
