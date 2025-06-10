# legal_ai_system/main.py
# FastAPI Backend for Legal AI System.

# Comprehensive API backend implementing:
# - JWT Authentication with role-based access control (currently mocked)
# - GraphQL for complex Knowledge Graph queries
# - WebSocket connections for real-time updates
# - RESTful endpoints for core operations
# - Integration with all Legal AI System components


import asyncio
import json
import logging
import os
import sys
import uuid  # For generating IDs
from collections import defaultdict

# import logging # Replaced by detailed_logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from jose import jwt

# third-party imports
try:
    import strawberry  # type: ignore
except ImportError:  # pragma: no cover - optional GraphQL dependency
    strawberry = None

try:
    import uvicorn
except ImportError:  # pragma: no cover - optional server dependency
    uvicorn = None

# Attempt to import project constants. If running in an isolated context where
# the package isn't installed, fall back to a minimal implementation so the
# script can still start without import errors.
try:
    from legal_ai_system.core.constants import Constants
except Exception:  # pragma: no cover - fallback for testing/archive usage

    class Constants:
        """Fallback constants used when the full package is unavailable."""

        DEBUG = True


# Assuming these will be structured correctly during refactoring
# Use absolute imports from the project root 'legal_ai_system'
try:
    from fastapi import Form  # Added Form
    from fastapi import (
        BackgroundTasks,
        Depends,
        FastAPI,
        File,
        HTTPException,
        UploadFile,
        WebSocket,
        WebSocketDisconnect,
        status,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import (  # Added HTMLResponse for root
        HTMLResponse,
        JSONResponse,
    )
    from fastapi.security import (
        HTTPAuthorizationCredentials,
        HTTPBearer,
    )
    from fastapi.staticfiles import StaticFiles
except ImportError:  # pragma: no cover - optional web dependency
    print("FastAPI not installed; exiting.")
    sys.exit(0)
from pydantic import BaseModel
from pydantic import Field as PydanticField  # Alias Field
from strawberry.fastapi import GraphQLRouter  # type: ignore
from strawberry.types import Info  # type: ignore

from config.settings import settings

# Attempt to import core services, with fallbacks for standalone running or partial setup
try:
    from legal_ai_system.config.settings import settings
    from legal_ai_system.core.detailed_logging import (
        LogCategory,
        get_detailed_logger,
    )
    SERVICES_AVAILABLE = True
except ImportError as e:
    # This fallback is for when main.py might be run before the full system is in place
    # or if there are circular dependencies during setup.
    print(
        f"WARNING: Core services import failed in main.py: {e}. API will run in a limited mock mode.",
        file=sys.stderr,
    )
    SERVICES_AVAILABLE = False
    ServiceContainer = None  # type: ignore
    SecurityManager = None  # type: ignore
    import logging

    class LogCategory(Enum):
        API = "API"

    def get_detailed_logger(name: str, category: LogCategory):
        return logging.getLogger(name)

    class AccessLevel(Enum):
        READ = "read"
        WRITE = "write"
        ADMIN = "admin"
        SUPER_ADMIN = "super_admin"

    class AuthUser:  # type: ignore
        def __init__(
            self,
            user_id: str,
            username: str,
            email: str,
            access_level: AccessLevel,
            last_login: Optional[datetime] = None,
            is_active: bool = True,
        ):
            self.user_id = user_id
            self.username = username
            self.email = email
            self.access_level = access_level
            self.last_login = last_login
            self.is_active = is_active

    class _SettingsFallback:
        frontend_dist_path = (
            Path(__file__).resolve().parent.parent / "frontend" / "dist"
        )

    settings = _SettingsFallback()


# Initialize logger for this module
main_api_logger = get_detailed_logger("FastAPI_Main", LogCategory.API)

# Global state (will be initialized in lifespan)
service_container_instance: Optional["ServiceContainer"] = None  # Renamed
security_manager_instance: Optional["SecurityManager"] = None  # Renamed
websocket_manager_instance: Optional["WebSocketManager"] = (
    None  # Renamed, forward declare WebSocketManager
)

# Workflow configuration storage
WORKFLOW_CONFIG_FILE = Path(settings.data_dir) / "workflow_configs.json"
workflow_configs: Dict[str, "WorkflowConfig"] = {}


def load_workflow_configs() -> None:
    """Load workflow presets from disk if available."""
    if WORKFLOW_CONFIG_FILE.exists():
        try:
            data = json.load(open(WORKFLOW_CONFIG_FILE, "r"))
            for item in data:
                workflow_configs[item["id"]] = WorkflowConfig(**item)
            main_api_logger.info(
                "Loaded workflow configurations",
                parameters={"count": len(workflow_configs)},
            )
        except Exception as e:  # pragma: no cover - startup resilience
            main_api_logger.error(
                "Failed to load workflow configurations.", exception=e
            )


def save_workflow_configs() -> None:
    """Persist workflow presets to disk."""
    try:
        WORKFLOW_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(WORKFLOW_CONFIG_FILE, "w") as f:
            json.dump(
                [cfg.model_dump() for cfg in workflow_configs.values()],
                f,
                default=str,
                indent=2,
            )
    except Exception as e:  # pragma: no cover - I/O failure shouldn't crash
        main_api_logger.error(
            "Failed to save workflow configurations.", exception=e
        )

# Workflow configuration storage
WORKFLOW_CONFIG_FILE = Path(settings.data_dir) / "workflow_configs.json"
workflow_configs: Dict[str, "WorkflowConfig"] = {}


def load_workflow_configs() -> None:
    """Load workflow presets from disk if available."""
    if WORKFLOW_CONFIG_FILE.exists():
        try:
            data = json.load(open(WORKFLOW_CONFIG_FILE, "r"))
            for item in data:
                workflow_configs[item["id"]] = WorkflowConfig(**item)
            main_api_logger.info(
                "Loaded workflow configurations",
                parameters={"count": len(workflow_configs)},
            )
        except Exception as e:  # pragma: no cover - startup resilience
            main_api_logger.error(
                "Failed to load workflow configurations.", exception=e
            )


def save_workflow_configs() -> None:
    """Persist workflow presets to disk."""
    try:
        WORKFLOW_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(WORKFLOW_CONFIG_FILE, "w") as f:
            json.dump(
                [cfg.model_dump() for cfg in workflow_configs.values()],
                f,
                default=str,
                indent=2,
            )
    except Exception as e:  # pragma: no cover - I/O failure shouldn't crash
        main_api_logger.error(
            "Failed to save workflow configurations.", exception=e
        )

# Workflow configuration storage
WORKFLOW_CONFIG_FILE = Path(settings.data_dir) / "workflow_configs.json"
workflow_configs: Dict[str, "WorkflowConfig"] = {}


def load_workflow_configs() -> None:
    """Load workflow presets from disk if available."""
    if WORKFLOW_CONFIG_FILE.exists():
        try:
            data = json.load(open(WORKFLOW_CONFIG_FILE, "r"))
            for item in data:
                workflow_configs[item["id"]] = WorkflowConfig(**item)
            main_api_logger.info(
                "Loaded workflow configurations",
                parameters={"count": len(workflow_configs)},
            )
        except Exception as e:  # pragma: no cover - startup resilience
            main_api_logger.error(
                "Failed to load workflow configurations.", exception=e
            )


def save_workflow_configs() -> None:
    """Persist workflow presets to disk."""
    try:
        WORKFLOW_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(WORKFLOW_CONFIG_FILE, "w") as f:
            json.dump(
                [cfg.model_dump() for cfg in workflow_configs.values()],
                f,
                default=str,
                indent=2,
            )
    except Exception as e:  # pragma: no cover - I/O failure shouldn't crash
        main_api_logger.error(
            "Failed to save workflow configurations.", exception=e
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global service_container_instance, security_manager_instance, websocket_manager_instance

    main_api_logger.info("🚀 Starting Legal AI System API lifespan...")

    if SERVICES_AVAILABLE and ServiceContainer is not None:
        try:
            # Factory function to build the service container
            from legal_ai_system.services.service_container import (
                create_service_container,
            )

            service_container_instance = (
                await create_service_container()
            )  # If it's async
            main_api_logger.info("✅ Service container initialized successfully.")
        except Exception as e:
            main_api_logger.error(
                "Failed to initialize service container.", exception=e
            )
            service_container_instance = None  # Ensure it's None if init fails
    else:
        main_api_logger.warning(
            "⚠️ ServiceContainer not available. API might run in a limited mode."
        )

    if SERVICES_AVAILABLE and SecurityManager is not None:
        try:
            # Configuration for SecurityManager should come from ConfigurationManager via service_container
            # For now, using placeholder values if service_container is not up.
            # In a full setup, ConfigurationManager would be a service itself.
            encryption_pwd = "default_strong_password_CHANGE_ME"
            allowed_dirs_list = [str(Path("./storage/documents/uploads").resolve())]

            if service_container_instance:
                config_manager = service_container_instance.get_service(
                    "configuration_manager"
                )
                if config_manager:
                    sec_config = config_manager.get_security_config()
                    # Assuming encryption_password might not be directly in settings for security reasons
                    # but fetched from a secure store or env var by SecurityManager itself.
                    # For allowed_directories, they should be part of the app's config.
                    allowed_dirs_list = sec_config.get(
                        "allowed_directories", allowed_dirs_list
                    )
                    # encryption_pwd might be handled internally by SecurityManager based on env vars

            security_manager_instance = SecurityManager(
                encryption_password=encryption_pwd,
                allowed_directories=allowed_dirs_list,
            )
            # Example user creation (in real app, this would be managed or seeded)
            if (
                security_manager_instance
                and not security_manager_instance.auth_manager.users
            ):
                security_manager_instance.auth_manager.create_user(
                    "demouser",
                    "demo@example.com",
                    "Password123!",
                    AccessLevel.ADMIN,
                )
                main_api_logger.info("Created default demo user.")

            main_api_logger.info("✅ Security manager initialized.")
        except Exception as e:
            main_api_logger.error("Failed to initialize SecurityManager.", exception=e)
            security_manager_instance = None
    else:
        main_api_logger.warning(
            "⚠️ SecurityManager not available. Authentication/Authorization will be bypassed."
        )

    # monitoring_task = asyncio.create_task(system_monitor_task())

    main_api_logger.info("✅ Legal AI System API started successfully via lifespan.")

    yield  # API is running

    main_api_logger.info("🛑 Shutting down Legal AI System API via lifespan...")
    # if monitoring_task: monitoring_task.cancel(); await asyncio.gather(monitoring_task, return_exceptions=True)
    if service_container_instance and hasattr(service_container_instance, "shutdown"):
        await service_container_instance.shutdown()
        main_api_logger.info("Service container shut down.")
    main_api_logger.info("Legal AI System API shutdown complete.")


app = FastAPI(
    title="Legal AI System API",
    description="Comprehensive API for Legal AI document processing and analysis",
    version=(getattr(getattr(Constants, "Version", None), "APP_VERSION", "2.0.1")),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "*",
    ],  # Added * for broader dev, tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the built frontend if available. Deployment can override the
# path via the FRONTEND_DIST_PATH environment variable.
frontend_dist = Path(settings.frontend_dist_path)
if frontend_dist.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(frontend_dist), html=True),
        name="frontend",
    )
else:
    main_api_logger.warning(
        "Frontend dist path not found; serving minimal root page.",
        extra={"path": str(frontend_dist)},
    )


# --- Security & Auth ---
# Mocked for now as per original file, to be integrated with SecurityManager
security_scheme = HTTPBearer()  # Renamed from security
JWT_SECRET_KEY = os.getenv(
    "LEGAL_AI_JWT_SECRET_KEY", "a_very_secret_key_for_jwt_replace_me_in_production"
)  # From env
JWT_ALGORITHM = "HS256"


class TokenData(BaseModel):  # For decoding token
    username: Optional[str] = None
    user_id: Optional[str] = None  # Added user_id


# Pydantic Models (Request/Response)
class TokenResponse(BaseModel):  # Renamed from Token
    access_token: str
    token_type: str
    # expires_in: int # Typically calculated by client or included in JWT 'exp'
    user: Dict[str, Any]  # User info to return


class LoginRequest(BaseModel):
    username: str
    password: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    size_bytes: int  # Renamed from size
    status: str
    message: Optional[str] = None  # Added message


class ProcessingRequest(BaseModel):
    # These should align with RealTimeAnalysisWorkflow options
    enable_ner: bool = PydanticField(
        True, description="Enable Named Entity Recognition"
    )
    enable_llm_extraction: bool = PydanticField(
        True, description="Enable LLM-based entity extraction"
    )
    # enable_targeted_prompting: bool = PydanticField(True) # This is part of hybrid_extractor config
    enable_confidence_calibration: bool = PydanticField(
        True, description="Enable confidence calibration"
    )
    confidence_threshold: float = PydanticField(
        0.7, ge=0.0, le=1.0, description="Confidence threshold for extractions"
    )
    # Add other relevant options from RealTimeAnalysisWorkflow if user-configurable


class WorkflowConfig(ProcessingRequest):
    """Preset configuration for a document processing workflow."""

    id: str = PydanticField(default_factory=lambda: uuid.uuid4().hex)
    name: str
    description: Optional[str] = None
    created_at: datetime = PydanticField(
        default_factory=lambda: datetime.now(tz=datetime.timezone.utc)
    )
    updated_at: datetime = PydanticField(
        default_factory=lambda: datetime.now(tz=datetime.timezone.utc)
    )


class WorkflowConfigCreate(ProcessingRequest):
    """Payload for creating a workflow preset."""

    name: str
    description: Optional[str] = None


class WorkflowConfigUpdate(ProcessingRequest):
    """Payload for updating a workflow preset."""

    name: Optional[str] = None
    description: Optional[str] = None


class WorkflowConfig(ProcessingRequest):
    """Preset configuration for a document processing workflow."""

    id: str = PydanticField(default_factory=lambda: uuid.uuid4().hex)
    name: str
    description: Optional[str] = None
    created_at: datetime = PydanticField(
        default_factory=lambda: datetime.now(tz=datetime.timezone.utc)
    )
    updated_at: datetime = PydanticField(
        default_factory=lambda: datetime.now(tz=datetime.timezone.utc)
    )


class WorkflowConfigCreate(ProcessingRequest):
    """Payload for creating a workflow preset."""

    name: str
    description: Optional[str] = None


class WorkflowConfigUpdate(ProcessingRequest):
    """Payload for updating a workflow preset."""

    name: Optional[str] = None
    description: Optional[str] = None


class WorkflowConfig(ProcessingRequest):
    """Preset configuration for a document processing workflow."""

    id: str = PydanticField(default_factory=lambda: uuid.uuid4().hex)
    name: str
    description: Optional[str] = None
    created_at: datetime = PydanticField(
        default_factory=lambda: datetime.now(tz=datetime.timezone.utc)
    )
    updated_at: datetime = PydanticField(
        default_factory=lambda: datetime.now(tz=datetime.timezone.utc)
    )


class WorkflowConfigCreate(ProcessingRequest):
    """Payload for creating a workflow preset."""

    name: str
    description: Optional[str] = None


class WorkflowConfigUpdate(ProcessingRequest):
    """Payload for updating a workflow preset."""

    name: Optional[str] = None
    description: Optional[str] = None


class DocumentStatusResponse(BaseModel):  # Renamed for clarity
    document_id: str
    status: str
    progress: float  # Changed to float for percentage 0.0-1.0
    stage: Optional[str] = None  # Current processing stage
    estimated_completion_sec: Optional[int] = None  # Renamed
    result_summary: Optional[Dict[str, Any]] = None  # If processing is complete


class ReviewDecisionRequest(BaseModel):
    item_id: str  # Renamed from entity_id for generality
    decision: str  # 'approve', 'reject', 'modify'
    modified_data: Optional[Dict[str, Any]] = None
    reviewer_notes: Optional[str] = None  # Added
    # confidence_adjustment: Optional[float] = None # This might be complex to expose directly


class SystemHealthResponse(BaseModel):
    overall_status: str  # Renamed from overall_health, e.g., "HEALTHY", "DEGRADED"
    services_status: Dict[str, Dict[str, Any]]  # Renamed from services
    performance_metrics_summary: Dict[str, float]  # Renamed from performance_metrics
    active_documents_count: int  # Renamed
    pending_reviews_count: int  # Renamed
    timestamp: str = PydanticField(
        default_factory=lambda: datetime.now(tz=datetime.timezone.utc).isoformat()
    )


# --- JWT Utilities & Auth Mock ---
# In a real app, these would use SecurityManager
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire_time = datetime.now(tz=datetime.timezone.utc) + (
        expires_delta or timedelta(hours=Constants.Time.SESSION_TIMEOUT_HOURS)
    )
    to_encode.update(
        {
            "exp": expire_time.timestamp(),
            "sub": data.get("username") or data.get("user_id"),
        }
    )  # 'sub' is standard
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


async def get_current_active_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
) -> AuthUser:
    """Mock: Validates token and returns user. Replace with real validation."""
    if not security_manager_instance:  # Bypass if security manager not up
        main_api_logger.warning(
            "Bypassing authentication: SecurityManager not available."
        )
        return AuthUser(
            user_id="mock_user",
            username="test_user",
            email="test@example.com",
            access_level=AccessLevel.ADMIN,
            last_login=datetime.now(tz=datetime.timezone.utc),
        )

    token = credentials.credentials
    user = security_manager_instance.auth_manager.validate_session(
        token
    )  # Assuming validate_session exists
    if not user:
        main_api_logger.warning(
            "Authentication failed: Invalid or expired token.",
            parameters={"token_preview": token[:10] + "..."},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        main_api_logger.warning(
            "Authentication failed: User inactive.",
            parameters={"user_id": user.user_id},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )
    return user


def require_permission(required_level: AccessLevel):
    """Dependency factory for permission checking."""

    async def permission_checker(
        current_user: AuthUser = Depends(get_current_active_user),
    ) -> AuthUser:
        if not security_manager_instance:  # Bypass if security manager not up
            main_api_logger.warning(
                "Bypassing permission check: SecurityManager not available."
            )
            return current_user

        if not security_manager_instance.auth_manager.check_permission(
            current_user, required_level
        ):
            main_api_logger.warning(
                "Permission denied.",
                parameters={
                    "user_id": current_user.user_id,
                    "required": required_level.value,
                    "actual": current_user.access_level.value,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Operation not permitted"
            )
        return current_user

    return permission_checker


# --- WebSocket Manager ---
class WebSocketManager:
    # ... (WebSocketManager from original file, with logging using main_api_logger.getChild("WebSocketManager"))
    # I will assume this class is defined as in the original main.py and add logging.
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}  # user_id -> WebSocket
        self.subscriptions: Dict[str, Set[str]] = defaultdict(
            set
        )  # user_id -> set of topics
        self.topic_subscribers: Dict[str, Set[str]] = defaultdict(
            set
        )  # topic -> set of user_ids
        self.logger = main_api_logger.getChild("WebSocketManager")  # Specific logger

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.logger.info(
            f"WebSocket connected.",
            parameters={"user_id": user_id, "client": str(websocket.client)},
        )
        await self.send_personal_message(
            {"type": "connection_ack", "status": "connected", "user_id": user_id},
            user_id,
        )

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

        topics_to_clean = list(self.subscriptions.pop(user_id, set()))
        for topic in topics_to_clean:
            if topic in self.topic_subscribers:
                self.topic_subscribers[topic].discard(user_id)
                if not self.topic_subscribers[topic]:  # Clean up empty topics
                    del self.topic_subscribers[topic]
        self.logger.info(f"WebSocket disconnected.", parameters={"user_id": user_id})

    async def send_personal_message(self, message: Dict[str, Any], user_id: str):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(
                    json.dumps(message, default=str)
                )
            except WebSocketDisconnect:
                self.logger.warning(
                    f"WebSocket already disconnected for user during send.",
                    parameters={"user_id": user_id},
                )
                self.disconnect(user_id)
            except Exception as e:
                self.logger.error(
                    f"Failed to send WebSocket message.",
                    parameters={"user_id": user_id},
                    exception=e,
                )
                self.disconnect(user_id)  # Assume connection is broken

    async def broadcast_to_topic(self, message: Dict[str, Any], topic: str):
        self.logger.debug(
            f"Broadcasting to topic.",
            parameters={
                "topic": topic,
                "num_subscribers": len(self.topic_subscribers.get(topic, [])),
            },
        )
        if topic in self.topic_subscribers:
            # Create a copy for safe iteration as disconnects might modify the set
            for user_id_subscriber in list(self.topic_subscribers[topic]):
                await self.send_personal_message(message, user_id_subscriber)

    async def subscribe_to_topic(self, user_id: str, topic: str):
        self.subscriptions[user_id].add(topic)
        self.topic_subscribers[topic].add(user_id)
        self.logger.info(
            f"User subscribed to topic.",
            parameters={"user_id": user_id, "topic": topic},
        )
        await self.send_personal_message(
            {"type": "subscription_ack", "topic": topic, "status": "subscribed"},
            user_id,
        )

    async def unsubscribe_from_topic(self, user_id: str, topic: str):
        self.subscriptions[user_id].discard(topic)
        if topic in self.topic_subscribers:
            self.topic_subscribers[topic].discard(user_id)
            if not self.topic_subscribers[topic]:
                del self.topic_subscribers[topic]
        self.logger.info(
            f"User unsubscribed from topic.",
            parameters={"user_id": user_id, "topic": topic},
        )
        await self.send_personal_message(
            {"type": "subscription_ack", "topic": topic, "status": "unsubscribed"},
            user_id,
        )


# --- GraphQL Schema Definitions ---
# Assuming these are defined as in the original main.py
# For brevity, I'll mock them up here. In a real refactor, these would be robust.
@strawberry.type
class GQLEntityType:  # Renamed to avoid conflict
    id: str
    name: str
    type: str  # Should map to EntityType enum ideally
    confidence: float = 1.0
    properties: Optional[strawberry.scalars.JSON] = None  # type: ignore
    # relationships: List["GQLRelationshipType"] # Forward reference


@strawberry.type
class GQLRelationshipType:  # Renamed
    id: str
    from_entity_id: str  # Renamed
    to_entity_id: str  # Renamed
    relationship_type: str
    confidence: float = 1.0
    properties: Optional[strawberry.scalars.JSON] = None  # type: ignore


@strawberry.type
class GQLDocumentType:  # Renamed
    id: str
    filename: str
    status: str
    progress: float  # Changed to float
    entities: Optional[List[GQLEntityType]] = None  # Made optional
    processing_time_sec: Optional[float] = None  # Renamed
    metadata: Optional[strawberry.scalars.JSON] = None  # type: ignore


@strawberry.type
class GQLReviewItemType:  # Renamed
    id: str
    item_text: str  # Renamed from entity_text
    item_type: str  # Renamed from entity_type
    confidence: float
    context_preview: str  # Renamed from context
    source_document_id: str  # Renamed
    # requires_review: bool # This is implicit if it's in the review queue


@strawberry.type
class GQLSystemStatusType:  # Renamed
    overall_status: str
    service_count: int
    healthy_services_count: int  # Renamed
    active_documents_count: int
    pending_reviews_count: int
    performance_metrics_summary: Optional[strawberry.scalars.JSON] = None  # type: ignore
    timestamp: str


# GraphQL Inputs (as per original)
@strawberry.input
class EntitySearchInput:
    query: str
    entity_types: Optional[List[str]] = None
    confidence_threshold: Optional[float] = PydanticField(None, ge=0.0, le=1.0)
    limit: Optional[int] = PydanticField(20, gt=0, le=100)


@strawberry.input
class GraphTraversalInput:
    entity_id: str
    max_depth: Optional[int] = PydanticField(2, ge=1, le=5)
    relationship_types: Optional[List[str]] = None
    # include_confidence_threshold: Optional[float] = None # This seems less common for traversal, more for search


# --- GraphQL Resolvers ---
@strawberry.type
class Query:
    @strawberry.field
    async def search_entities(
        self, search_input: EntitySearchInput, info: Info
    ) -> List[GQLEntityType]:
        main_api_logger.info(
            "GraphQL: search_entities called", parameters=search_input.__dict__
        )
        if not service_container_instance:
            return []
        kg_manager = service_container_instance.get_service("knowledge_graph_manager")
        if not kg_manager:
            return []
        # Adapt to KnowledgeGraphManager's search method
        # results = await kg_manager.find_entities(...)
        return [
            GQLEntityType(id="e1", name="Mock Entity", type="PERSON", confidence=0.9)
        ]  # Mock

    @strawberry.field
    async def get_document_status_gql(
        self, document_id: str, info: Info
    ) -> Optional[GQLDocumentType]:  # Renamed
        main_api_logger.info(
            "GraphQL: get_document_status called",
            parameters={"document_id": document_id},
        )
        # Mock implementation
        # In real version, fetch from a document status store
        # status_data = await get_document_status_rest(document_id) # Call REST version or service
        return GQLDocumentType(
            id=document_id, filename="mock.pdf", status="processing", progress=0.75
        )

    @strawberry.field
    async def system_status_gql(self, info: Info) -> GQLSystemStatusType:  # Renamed
        main_api_logger.info("GraphQL: system_status called")
        if not service_container_instance:
            return GQLSystemStatusType(
                overall_status="ERROR",
                service_count=0,
                healthy_services_count=0,
                active_documents_count=0,
                pending_reviews_count=0,
                timestamp=datetime.now().isoformat(),
            )
        # This should call a method on service_container_instance or a dedicated status service
        # status_data = await service_container_instance.get_system_status_summary()
        return GQLSystemStatusType(
            overall_status="HEALTHY",
            service_count=5,
            healthy_services_count=5,
            active_documents_count=2,
            pending_reviews_count=1,
            timestamp=datetime.now().isoformat(),
        )  # Mock


# Create GraphQL schema & router
gql_schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app_router = GraphQLRouter(gql_schema, graphiql=True)  # Enable GraphiQL for dev
app.include_router(graphql_app_router, prefix="/graphql")


# --- REST API Endpoints ---
if not Path(settings.frontend_dist_path).is_dir():

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def api_root():
        """Fallback landing page shown when frontend assets are missing."""
        return """
        <html>
            <head><title>Legal AI System API</title></head>
            <body>
                <h1>Welcome to the Legal AI System API</h1>
                <p>This is the central backend for all Legal AI operations.</p>
                <ul>
                    <li><a href="/docs">API Documentation (Swagger UI)</a></li>
                    <li><a href="/redoc">Alternative API Documentation (ReDoc)</a></li>
                    <li><a href="/graphql">GraphQL Endpoint (GraphiQL)</a></li>
                </ul>
            </body>
        </html>
        """


@app.post("/api/v1/auth/token", response_model=TokenResponse)
async def login_for_access_token(
    form_data: LoginRequest,
):  # Changed to form_data for clarity
    main_api_logger.info("Login attempt", parameters={"username": form_data.username})
    if not security_manager_instance:  # Mock if no security manager
        main_api_logger.warning(
            "Auth bypassed: SecurityManager not available. Issuing mock token."
        )
        mock_user_info = {
            "user_id": "mock_user_id",
            "username": form_data.username,
            "email": "mock@example.com",
            "access_level": "admin",
        }
        access_token = create_access_token(
            data={
                "sub": form_data.username,
                "user_id": "mock_user_id",
                "roles": ["admin"],
            }
        )
        return TokenResponse(
            access_token=access_token, token_type="bearer", user=mock_user_info
        )

    session_token = security_manager_instance.auth_manager.authenticate(
        form_data.username, form_data.password
    )
    if not session_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    user_obj = next(
        (
            u
            for u in security_manager_instance.auth_manager.users.values()
            if u.username == form_data.username
        ),
        None,
    )
    user_info = (
        {
            "user_id": user_obj.user_id,
            "username": user_obj.username,
            "email": user_obj.email,
            "access_level": user_obj.access_level.value,
        }
        if user_obj
        else {}
    )

    # Create a JWT token from the session token or directly with user info
    # For simplicity here, we'll create a JWT directly. In a more stateful system, session_token might be the JWT.
    jwt_access_token = create_access_token(
        data={
            "user_id": user_obj.user_id if user_obj else "unknown",
            "username": form_data.username,
            "roles": [user_obj.access_level.value if user_obj else "read"],
        }
    )
    return TokenResponse(
        access_token=jwt_access_token, token_type="bearer", user=user_info
    )


@app.get(
    "/api/v1/auth/me", response_model=Dict[str, Any]
)  # Define a Pydantic model for UserInfo for better typing
async def read_users_me(current_user: AuthUser = Depends(get_current_active_user)):
    main_api_logger.info(
        "Fetching current user info", parameters={"user_id": current_user.user_id}
    )
    return {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "email": current_user.email,
        "access_level": (
            current_user.access_level.value
            if isinstance(current_user.access_level, Enum)
            else current_user.access_level
        ),
        "last_login": (
            current_user.last_login.isoformat() if current_user.last_login else None
        ),
        "is_active": current_user.is_active,
    }


@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document_rest(  # Renamed to avoid conflict
    file: UploadFile = File(...),
    # current_user: AuthUser = Depends(require_permission(AccessLevel.WRITE)) # Auth re-enabled
    # For now, removing auth dependency for ease of testing if SecurityManager isn't fully up
):
    main_api_logger.info(
        "Document upload request received.",
        parameters={"filename": file.filename, "content_type": file.content_type},
    )
    # For MVP, save locally. In prod, use secure storage (e.g., S3) via a storage service.
    # This path should come from ConfigurationManager.
    upload_dir = Path("./storage/documents/uploads_api")
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_filename = "".join(
        c if c.isalnum() or c in [".", "-", "_"] else "_"
        for c in file.filename or "unknown_file"
    )
    timestamp = datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d%H%M%S%f")
    unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{safe_filename}"
    file_path = upload_dir / unique_filename

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        doc_size = len(content)
        # Generate a unique document ID (e.g., using UUID or a hash of content+timestamp)
        document_id = f"doc_{uuid.uuid4().hex}"

        # Here, you would typically store metadata about the uploaded document in a database.
        # For example, linking document_id to file_path, user_id, upload_time, status='uploaded'.
        # This is mocked in the original `minimal_api.py`.

        main_api_logger.info(
            "Document uploaded successfully.",
            parameters={
                "document_id": document_id,
                "filename": unique_filename,
                "path": str(file_path),
                "size_bytes": doc_size,
            },
        )

        return DocumentUploadResponse(
            document_id=document_id,
            filename=unique_filename,  # Return the unique, safe filename
            size_bytes=doc_size,
            status="uploaded",
            message="Document uploaded successfully. Ready for processing.",
        )
    except Exception as e:
        main_api_logger.error(
            "Document upload failed.",
            parameters={"filename": file.filename},
            exception=e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}",
        )


@app.post(
    "/api/v1/documents/{document_id}/process", status_code=status.HTTP_202_ACCEPTED
)
async def process_document_rest(  # Renamed
    document_id: str,
    processing_request: ProcessingRequest,  # Use Pydantic model for request body
    background_tasks: BackgroundTasks,
    # current_user: AuthUser = Depends(require_permission(AccessLevel.WRITE)) # Auth
):
    main_api_logger.info(
        "Request to process document.",
        parameters={
            "document_id": document_id,
            "options": processing_request.model_dump(),
        },
    )

    # Validate document_id format or existence in DB (if applicable)
    # For now, assume document_id refers to a previously uploaded file.
    # The actual file path needs to be retrieved based on document_id.
    # This is a placeholder for that logic.

    # This path needs to be robustly determined from document_id, e.g., from a DB lookup
    # For now, construct a plausible path based on the upload logic. This is NOT PRODUCTION READY.
    # This assumes the document_id given here might be the unique_filename from upload.
    # A better system would have a DB mapping doc_id to its stored path.

    # Try to find the file. This is a simplification.
    # In a real system, you'd look up 'document_id' in a database to get its stored path.
    upload_dir = Path("./storage/documents/uploads_api")
    # Attempt to find a file that might correspond to this document_id.
    # This is a simplified search. A real system would use a DB.
    possible_files = list(
        upload_dir.glob(
            f"*_{document_id.split('_')[-1]}*"
            if "_" in document_id
            else f"*{document_id}*"
        )
    )

    if not possible_files:
        # If not found by original name part, try to see if document_id is the full unique name
        exact_file_path = upload_dir / document_id
        if exact_file_path.exists():
            document_file_path_str = str(exact_file_path)
        else:
            main_api_logger.error(
                f"Document file for ID '{document_id}' not found in upload directory."
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found or path cannot be resolved.",
            )
    else:
        document_file_path_str = str(
            possible_files[0]
        )  # Take the first match for simplicity

    workflow = await service_container_instance.get_service(
        "realtime_analysis_workflow"
    )
    await service_container_instance.update_workflow_config(
        processing_request.model_dump()
    )

    # user_id_for_task = current_user.user_id
    user_id_for_task = "mock_user_for_processing"  # Placeholder if auth is off

    background_tasks.add_task(
        process_document_background_task,  # Renamed
        document_id,  # Pass the conceptual document_id
        document_file_path_str,  # Pass the actual file path for processing
        processing_request,
        user_id_for_task,  # Pass user ID for auditing/context
    )

    main_api_logger.info(
        "Document processing task added to background.",
        parameters={"document_id": document_id},
    )
    return {
        "message": "Document processing started in background.",
        "document_id": document_id,
    }


@app.get(
    "/api/v1/documents/{document_id}/status", response_model=DocumentStatusResponse
)
async def get_document_status_rest(  # Renamed
    document_id: str,
    # current_user: AuthUser = Depends(require_permission(AccessLevel.READ)) # Auth
):
    main_api_logger.debug(
        "Request for document status.", parameters={"document_id": document_id}
    )
    # This should query a persistent store or an in-memory state manager for the actual status.
    # For now, returning a mock status.
    # Example: status_info = await service_container_instance.get_service("workflow_state_manager").get_status(document_id)

    # Mock status
    # In a real system, this would come from a database or a state manager.
    # Check if the document ID is in a (hypothetical) processing state tracker
    # global_processing_states is a placeholder for actual state management
    if "global_processing_states" in globals() and document_id in global_processing_states:  # type: ignore
        state = global_processing_states[document_id]  # type: ignore
        return DocumentStatusResponse(
            document_id=document_id,
            status=state.get("status", "unknown"),
            progress=state.get("progress", 0.0),
            stage=state.get("stage"),
            # result_summary=state.get("result_summary") # If available
        )

    # If not actively processing or no info, assume pending or check DB
    # This is highly dependent on how processing states are stored.
    # Let's return a generic "unknown" or "pending" if not found in active states.
    main_api_logger.warning(
        "Document status not found in active processing. Returning placeholder.",
        parameters={"document_id": document_id},
    )
    return DocumentStatusResponse(
        document_id=document_id, status="pending_or_unknown", progress=0.0
    )


# ----- Workflow Config Endpoints -----

@app.get("/api/v1/workflows", response_model=List[WorkflowConfig])
async def list_workflow_configs():
    """List all saved workflow presets."""
    return list(workflow_configs.values())


@app.post(
    "/api/v1/workflows",
    response_model=WorkflowConfig,
    status_code=status.HTTP_201_CREATED,
)
async def create_workflow_config(config: WorkflowConfigCreate):
    new_cfg = WorkflowConfig(**config.model_dump())
    workflow_configs[new_cfg.id] = new_cfg
    save_workflow_configs()
    return new_cfg


@app.put("/api/v1/workflows/{config_id}", response_model=WorkflowConfig)
async def update_workflow_config(config_id: str, update: WorkflowConfigUpdate):
    existing = workflow_configs.get(config_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Workflow config not found")
    update_data = update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(existing, key, value)
    existing.updated_at = datetime.now(tz=datetime.timezone.utc)
    workflow_configs[config_id] = existing
    save_workflow_configs()
    return existing


@app.delete(
    "/api/v1/workflows/{config_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_workflow_config(config_id: str):
    if config_id in workflow_configs:
        del workflow_configs[config_id]
        save_workflow_configs()
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)
    raise HTTPException(status_code=404, detail="Workflow config not found")


# ----- Workflow Config Endpoints -----

@app.get("/api/v1/workflows", response_model=List[WorkflowConfig])
async def list_workflow_configs():
    """List all saved workflow presets."""
    return list(workflow_configs.values())


@app.post(
    "/api/v1/workflows",
    response_model=WorkflowConfig,
    status_code=status.HTTP_201_CREATED,
)
async def create_workflow_config(config: WorkflowConfigCreate):
    new_cfg = WorkflowConfig(**config.model_dump())
    workflow_configs[new_cfg.id] = new_cfg
    save_workflow_configs()
    return new_cfg


@app.put("/api/v1/workflows/{config_id}", response_model=WorkflowConfig)
async def update_workflow_config(config_id: str, update: WorkflowConfigUpdate):
    existing = workflow_configs.get(config_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Workflow config not found")
    update_data = update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(existing, key, value)
    existing.updated_at = datetime.now(tz=datetime.timezone.utc)
    workflow_configs[config_id] = existing
    save_workflow_configs()
    return existing


@app.delete(
    "/api/v1/workflows/{config_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_workflow_config(config_id: str):
    if config_id in workflow_configs:
        del workflow_configs[config_id]
        save_workflow_configs()
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)
    raise HTTPException(status_code=404, detail="Workflow config not found")


# ----- Workflow Config Endpoints -----

@app.get("/api/v1/workflows", response_model=List[WorkflowConfig])
async def list_workflow_configs():
    """List all saved workflow presets."""
    return list(workflow_configs.values())


@app.post(
    "/api/v1/workflows",
    response_model=WorkflowConfig,
    status_code=status.HTTP_201_CREATED,
)
async def create_workflow_config(config: WorkflowConfigCreate):
    new_cfg = WorkflowConfig(**config.model_dump())
    workflow_configs[new_cfg.id] = new_cfg
    save_workflow_configs()
    return new_cfg


@app.put("/api/v1/workflows/{config_id}", response_model=WorkflowConfig)
async def update_workflow_config(config_id: str, update: WorkflowConfigUpdate):
    existing = workflow_configs.get(config_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Workflow config not found")
    update_data = update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(existing, key, value)
    existing.updated_at = datetime.now(tz=datetime.timezone.utc)
    workflow_configs[config_id] = existing
    save_workflow_configs()
    return existing


@app.delete(
    "/api/v1/workflows/{config_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_workflow_config(config_id: str):
    if config_id in workflow_configs:
        del workflow_configs[config_id]
        save_workflow_configs()
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)
    raise HTTPException(status_code=404, detail="Workflow config not found")


@app.get("/api/v1/system/health", response_model=SystemHealthResponse)
async def get_system_health_rest(  # Renamed
    # current_user: AuthUser = Depends(require_permission(AccessLevel.READ)) # Auth
):
    main_api_logger.info("System health check requested.")
    if not service_container_instance or not hasattr(
        service_container_instance, "get_system_health_summary"
    ):
        main_api_logger.error(
            "Cannot get system health: ServiceContainer not available or method missing."
        )
        # Return a degraded status if core components are missing
        return SystemHealthResponse(
            overall_status="ERROR",
            services_status={
                "manager": {
                    "status": "unavailable",
                    "details": "Service container not initialized",
                }
            },
            performance_metrics_summary={},
            active_documents_count=0,
            pending_reviews_count=0,
            timestamp=datetime.now(tz=datetime.timezone.utc).isoformat(),
        )

    try:
        # This method should be on ServiceContainer or a dedicated HealthService
        health_summary = await service_container_instance.get_system_health_summary()

        return SystemHealthResponse(
            overall_status=health_summary.get("overall_status", "DEGRADED"),
            services_status=health_summary.get("services_status", {}),
            performance_metrics_summary=health_summary.get(
                "performance_metrics_summary", {}
            ),
            active_documents_count=health_summary.get("active_documents_count", 0),
            pending_reviews_count=health_summary.get("pending_reviews_count", 0),
            timestamp=health_summary.get(
                "timestamp", datetime.now(tz=datetime.timezone.utc).isoformat()
            ),
        )
    except Exception as e:
        main_api_logger.error("Failed to get system health.", exception=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}",
        )


@app.post("/api/v1/calibration/review", status_code=status.HTTP_200_OK)
async def submit_review_decision_rest(  # Renamed
    review_request: ReviewDecisionRequest,
    # current_user: AuthUser = Depends(require_permission(AccessLevel.WRITE)) # Auth
):
    main_api_logger.info(
        "Review decision submitted.", parameters=review_request.model_dump()
    )
    if not service_container_instance:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Review service not configured.",
        )

    review_service = service_container_instance.get_service(
        "reviewable_memory"
    )  # Or 'confidence_calibration_manager'
    if not review_service or not hasattr(review_service, "submit_review_decision"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Review submission component not available.",
        )

    try:
        # Assuming submit_review_decision on the service takes similar params
        # success = await review_service.submit_review_decision(
        #     item_id=review_request.item_id,
        #     decision_status=review_request.decision, # Map string to enum if needed by service
        #     modified_content=review_request.modified_data,
        #     notes=review_request.reviewer_notes
        # )
        # Mocking success for now
        success = True

        if success:
            if websocket_manager_instance:  # Check if initialized
                await websocket_manager_instance.broadcast_to_topic(
                    {
                        "type": "review_processed",  # Standardized event type
                        "item_id": review_request.item_id,
                        "decision": review_request.decision,
                        # "user": current_user.username, # If auth is on
                        "user": "mock_reviewer",
                        "timestamp": datetime.now(tz=datetime.timezone.utc).isoformat(),
                    },
                    "calibration_updates",
                )  # Specific topic for calibration
            return {"status": "review_processed", "item_id": review_request.item_id}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to process review decision.",
            )

    except Exception as e:
        main_api_logger.error(
            "Review decision submission failed.",
            parameters={"item_id": review_request.item_id},
            exception=e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Review processing failed: {str(e)}",
        )


# --- WebSocket Endpoint ---
@app.websocket(
    "/ws/{user_id}"
)  # Consider making user_id part of authenticated token if not already
async def websocket_endpoint_route(websocket: WebSocket, user_id: str):  # Renamed
    # user_id from path might be for initial connection, but real user_id should come from an auth token over WS
    # For now, we'll use the path user_id.
    if not websocket_manager_instance:
        main_api_logger.error(
            "WebSocket connection attempt failed: WebSocketManager not initialized."
        )
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    await websocket_manager_instance.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            main_api_logger.debug(
                "WebSocket message received.",
                parameters={"user_id": user_id, "message_type": message.get("type")},
            )

            msg_type = message.get("type")
            if msg_type == "subscribe":
                topic = message.get("topic")
                if topic:
                    await websocket_manager_instance.subscribe_to_topic(user_id, topic)
            elif msg_type == "unsubscribe":
                topic = message.get("topic")
                if topic:
                    await websocket_manager_instance.unsubscribe_from_topic(
                        user_id, topic
                    )
            elif msg_type == "ping":
                await websocket_manager_instance.send_personal_message(
                    {"type": "pong", "timestamp": datetime.now().isoformat()}, user_id
                )
            # Add more message type handlers as needed
            else:
                main_api_logger.warning(
                    "Unknown WebSocket message type received.",
                    parameters={"user_id": user_id, "message": message},
                )

    except WebSocketDisconnect:
        main_api_logger.info(
            f"WebSocket client disconnected.", parameters={"user_id": user_id}
        )
    except Exception as e:
        main_api_logger.error(
            f"WebSocket error.", parameters={"user_id": user_id}, exception=e
        )
    finally:
        if websocket_manager_instance:
            websocket_manager_instance.disconnect(user_id)


# --- Background Tasks ---
# Placeholder for actual state tracking if not using a dedicated state manager service
global_processing_states: Dict[str, Dict[str, Any]] = {}


async def process_document_background_task(  # Renamed
    document_id: str,
    document_file_path: str,  # Actual path to the file
    processing_request_model: ProcessingRequest,  # Use the Pydantic model
    requesting_user_id: str,  # For audit
):
    main_api_logger.info(
        "Background task started for document processing.",
        parameters={
            "document_id": document_id,
            "file_path": document_file_path,
            "user_id": requesting_user_id,
        },
    )

    global_processing_states[document_id] = {
        "status": "starting",
        "progress": 0.01,
        "stage": "Initializing",
    }

        # Define progress callback for WebSocket
        async def ws_progress_callback(
            stage: str,
            progress_percent: float,
            details: Optional[Dict[str, Any]] = None,
        ):
            global_processing_states[document_id].update(
                {
                    "status": "processing",
                    "progress": progress_percent / 100.0,
                    "stage": stage,
                    "details": details,
                }
            )
            if websocket_manager_instance:
                await websocket_manager_instance.broadcast_to_topic(
                    {
                        "type": "processing_progress",
                        "document_id": document_id,
                        "progress": progress_percent,  # Send as 0-100
                        "stage": stage,
                        "details": details or {},
                    },
                    f"document_updates_{document_id}",
                )

        workflow.register_progress_callback(
            ws_progress_callback
        )  # Assuming workflow supports this

        # Execute the workflow
        global_processing_states[document_id].update(
            {
                "status": "completed",
                "progress": 1.0,
                "stage": "Complete",
                "result_summary": {  # Example summary
                    "entities_found": (
                        len(analysis_result.hybrid_extraction.validated_entities)
                        if analysis_result.hybrid_extraction
                        else 0
                    ),
                    "total_time_sec": analysis_result.total_processing_time,
                },
            }
        )

        if websocket_manager_instance:
            await websocket_manager_instance.broadcast_to_topic(
                {
                    "type": "processing_complete",
                    "document_id": document_id,
                    "result": analysis_result.to_dict(),  # Send full or summarized result
                },
                f"document_updates_{document_id}",
            )

        main_api_logger.info(
            "Document processing background task finished successfully.",
            parameters={
                "document_id": document_id,
                "total_time_sec": analysis_result.total_processing_time,
            },
        )

    except Exception as e:
        main_api_logger.error(
            f"Background processing failed for document.",
            parameters={"document_id": document_id, "file_path": document_file_path},
            exception=e,
        )
        global_processing_states[document_id].update(
            {
                "status": "failed",
                "progress": global_processing_states[document_id].get("progress", 0.0),
                "error": str(e),
            }
        )
        if websocket_manager_instance:
            await websocket_manager_instance.broadcast_to_topic(
                {
                    "type": "processing_error",
                    "document_id": document_id,
                    "error": str(e),
                },
                f"document_updates_{document_id}",
            )


if __name__ == "__main__":
    # This allows running the FastAPI app directly using `python main.py`
    # Ensure detailed_logging is configured before uvicorn starts for its logs to be captured.
    # The lifespan function handles most initialization.

    # Basic logging setup if detailed_logging hasn't been configured by a higher-level entry point
    if (
        not main_api_logger.logger.hasHandlers()
    ):  # Check if our specific logger got handlers from detailed_logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        main_api_logger.info(
            "Using basicConfig for main_api_logger as detailed_logging handlers were not found."
        )

    main_api_logger.info("Starting FastAPI server directly via uvicorn...")
    uvicorn.run(
        "main:app",  # Points to this file (main.py) and the app instance
        host=os.getenv("LEGAL_AI_API_HOST", "0.0.0.0"),
        port=int(os.getenv("LEGAL_AI_API_PORT", "8000")),
        reload=True,  # Enable reload for development
        log_level="info",  # Uvicorn's own log level
    )
