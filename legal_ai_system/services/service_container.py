# legal_ai_system/core/service_container.py
"""
Service Container for Dependency Injection and Lifecycle Management.

Manages the creation, retrieval, initialization, and shutdown of all
core services and agents within the Legal AI System.
"""

import asyncio
from typing import Dict, Any, Optional, Callable, Awaitable, List, TYPE_CHECKING
from enum import Enum
from datetime import datetime, timezone
import os

import sys
from pathlib import Path

# Add project root to path for absolute imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if TYPE_CHECKING:
    from legal_ai_system.core.detailed_logging import (
        DetailedLogger,
        get_detailed_logger,
        LogCategory,
        detailed_log_function,
    )
    from legal_ai_system.core.unified_exceptions import (
        ConfigurationError,
        SystemInitializationError,
    )
    from legal_ai_system.services.security_manager import AuthenticationManager
else:
    # Use absolute imports with fallback for runtime
    try:
        from legal_ai_system.core.detailed_logging import (
            DetailedLogger,
            get_detailed_logger,
            LogCategory,
            detailed_log_function,
        )
        from legal_ai_system.core.unified_exceptions import (
            ConfigurationError,
            SystemInitializationError,
        )
        from legal_ai_system.services.security_manager import AuthenticationManager
    except ImportError:
        # Fallback for relative imports when running within package
        try:
            from ..core.detailed_logging import (
                DetailedLogger,
                get_detailed_logger,
                LogCategory,
                detailed_log_function,
            )
            from ..core.unified_exceptions import (
                ConfigurationError,
                SystemInitializationError,
            )
            from .security_manager import AuthenticationManager
        except ImportError:
            # Final fallback - create minimal classes to keep runtime working
            import logging

            class LogCategory(Enum):
                SYSTEM = "SYSTEM"

            class DetailedLogger(logging.Logger):
                pass

            def get_detailed_logger(name: str, category: LogCategory) -> DetailedLogger:  # type: ignore
                return logging.getLogger(name)  # type: ignore

            def detailed_log_function(category: LogCategory):
                def decorator(func):
                    return func

                return decorator

            class ConfigurationError(Exception):
                pass

            class SystemInitializationError(Exception):
                pass

            class AuthenticationManager:
                pass


# Initialize logger for this module
service_container_logger: DetailedLogger = get_detailed_logger(
    "ServiceContainer", LogCategory.SYSTEM
)


class ServiceLifecycleState(Enum):
    """States of a service within the container."""

    REGISTERED = "registered"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    ERROR = "error_during_init"
    SHUTDOWN = "shutdown"


class ServiceContainer:
    """
    Manages the lifecycle and provides access to system services and agents.
    """

    @detailed_log_function(LogCategory.SYSTEM)
    def __init__(self):
        service_container_logger.info("Initializing ServiceContainer.")
        self._services: Dict[str, Any] = {}
        # Stores factory metadata for lazy service creation
        self._service_factories: Dict[str, Dict[str, Any]] = {}
        self._service_states: Dict[str, ServiceLifecycleState] = {}
        self._initialization_order: List[str] = []  # To manage dependencies during init
        self._shutdown_order: List[str] = []  # Reverse of init order
        self._async_tasks: List[asyncio.Task] = (
            []
        )  # For background tasks started by services
        self._lock = (
            asyncio.Lock()
        )  # For thread-safe registration and retrieval if needed (though primarily async)
        service_container_logger.info("ServiceContainer instance created.")

    @detailed_log_function(LogCategory.SYSTEM)
    async def register_service(
        self,
        name: str,
        instance: Optional[Any] = None,
        factory: Optional[Callable[..., Any]] = None,
        is_async_factory: bool = False,
        depends_on: Optional[List[str]] = None,
        config_key: Optional[str] = None,  # Key to fetch config for this service
        **factory_kwargs: Any,
    ):
        """
        Registers a service instance or a factory to create it.
        If a factory is provided, the service will be created on first get() or during initialize_all().
        """
        async with self._lock:
            if name in self._services or name in self._service_factories:
                service_container_logger.warning(
                    f"Service '{name}' already registered. Overwriting.",
                    parameters={"name": name},
                )

            if instance is not None:
                self._services[name] = instance
                self._service_states[name] = (
                    ServiceLifecycleState.INITIALIZED
                )  # Assume pre-initialized if instance given
                self._initialization_order.append(name)  # Add to init order
                service_container_logger.info(
                    f"Service instance registered.",
                    parameters={"name": name, "type": type(instance).__name__},
                )
            elif factory is not None:
                self._service_factories[name] = {
                    "factory": factory,
                    "is_async": is_async_factory,
                    "depends_on": depends_on or [],
                    "config_key": config_key,
                    "kwargs": factory_kwargs,
                }
                self._service_states[name] = ServiceLifecycleState.REGISTERED
                self._initialization_order.append(name)  # Add to init order
                service_container_logger.info(
                    f"Service factory registered.", parameters={"name": name}
                )
            else:
                msg = "Either an instance or a factory must be provided to register_service."
                service_container_logger.error(msg, parameters={"name": name})
                raise ValueError(msg)

    @detailed_log_function(LogCategory.SYSTEM)
    async def get_service(self, name: str) -> Any:
        """Retrieves a service instance, creating it via factory if necessary."""
        async with self._lock:
            if name not in self._services:
                if name not in self._service_factories:
                    service_container_logger.error(
                        f"Service not found.", parameters={"name": name}
                    )
                    raise ConfigurationError(
                        f"Service '{name}' not found in container."
                    )

                # Create service from factory
                factory_info = self._service_factories[name]
                service_container_logger.info(
                    f"Creating service '{name}' from factory."
                )

                # Resolve dependencies first
                for dep_name in factory_info["depends_on"]:
                    if (
                        dep_name not in self._services
                    ):  # Ensure dependency is initialized
                        await self.get_service(dep_name)

                # Get config for the service if config_key is provided
                service_config = {}
                if factory_info["config_key"]:
                    config_manager = self._services.get(
                        "configuration_manager"
                    )  # Assume CM is registered
                    if config_manager and hasattr(config_manager, "get"):
                        service_config = config_manager.get(
                            factory_info["config_key"], {}
                        )
                    else:
                        service_container_logger.warning(
                            f"ConfigurationManager not found or 'get' method missing. Cannot load config for service '{name}'."
                        )

                # Merge factory_kwargs with loaded service_config (kwargs take precedence)
                final_kwargs = {**service_config, **factory_info["kwargs"]}

                try:
                    if factory_info["is_async"]:
                        instance = await factory_info["factory"](
                            self, **final_kwargs
                        )  # Pass container and merged kwargs
                    else:
                        instance = factory_info["factory"](self, **final_kwargs)

                    self._services[name] = instance
                    # Initialization is now handled by initialize_all_services or explicitly
                    # self._service_states[name] = ServiceLifecycleState.INITIALIZING
                    # if hasattr(instance, 'initialize_service'):
                    #     await instance.initialize_service()
                    # elif hasattr(instance, 'initialize'):
                    #     await instance.initialize()
                    # self._service_states[name] = ServiceLifecycleState.INITIALIZED

                    service_container_logger.info(
                        f"Service '{name}' created and cached."
                    )
                except Exception as e:
                    self._service_states[name] = ServiceLifecycleState.ERROR
                    service_container_logger.critical(
                        f"Failed to create service '{name}' from factory.", exception=e
                    )
                    raise SystemInitializationError(
                        f"Failed to create service '{name}'", cause=e
                    )

            return self._services[name]

    @detailed_log_function(LogCategory.SYSTEM)
    async def initialize_all_services(self):
        """Initializes all registered services that have an 'initialize_service' or 'initialize' method."""
        service_container_logger.info("Initializing all registered services...")
        # Sort services by dependency order if complex dependencies exist (topological sort)
        # For now, using registration order.

        # First, instantiate all factory-based services that haven't been created yet
        for name in list(
            self._service_factories.keys()
        ):  # Iterate on copy as get_service modifies _services
            if name not in self._services:
                await self.get_service(name)  # This will create it

        # Now initialize them
        for name in self._initialization_order:
            if (
                name in self._services
                and self._service_states.get(name) != ServiceLifecycleState.INITIALIZED
            ):
                instance = self._services[name]
                init_method_name = None
                if hasattr(instance, "initialize_service"):
                    init_method_name = "initialize_service"
                elif hasattr(instance, "initialize"):
                    init_method_name = "initialize"

                if init_method_name:
                    self._service_states[name] = ServiceLifecycleState.INITIALIZING
                    service_container_logger.info(f"Initializing service '{name}'...")
                    try:
                        init_method = getattr(instance, init_method_name)
                        if asyncio.iscoroutinefunction(init_method):
                            await init_method()
                        else:  # Run sync init in executor if it's potentially blocking
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, init_method)
                        self._service_states[name] = ServiceLifecycleState.INITIALIZED
                        service_container_logger.info(
                            f"Service '{name}' initialized successfully."
                        )
                    except Exception as e:
                        self._service_states[name] = ServiceLifecycleState.ERROR
                        service_container_logger.error(
                            f"Failed to initialize service '{name}'.", exception=e
                        )
                        # Decide: re-raise, or just log and continue?
                        # raise SystemInitializationError(f"Failed to initialize service '{name}'", cause=e)
                else:  # No init method, assume it's ready or initialized in constructor
                    self._service_states[name] = ServiceLifecycleState.INITIALIZED
                    service_container_logger.debug(
                        f"Service '{name}' has no specific initialize method, assumed ready."
                    )
        service_container_logger.info("All services initialization process completed.")

    @detailed_log_function(LogCategory.SYSTEM)
    async def shutdown_all_services(self):
        """Shuts down all registered services that have a 'shutdown' or 'close' method."""
        service_container_logger.info("Shutting down all registered services...")

        # Cancel any background tasks
        for task in self._async_tasks:
            if not task.done():
                task.cancel()
        if self._async_tasks:
            await asyncio.gather(*self._async_tasks, return_exceptions=True)
            service_container_logger.info(
                f"Cancelled {len(self._async_tasks)} background tasks."
            )

        # Shutdown in reverse order of initialization
        shutdown_order_actual = [
            name
            for name in reversed(self._initialization_order)
            if name in self._services
        ]

        for name in shutdown_order_actual:
            instance = self._services.get(name)
            if instance:
                shutdown_method_name = None
                if hasattr(instance, "shutdown"):
                    shutdown_method_name = "shutdown"
                elif hasattr(instance, "close"):
                    shutdown_method_name = "close"

                if shutdown_method_name:
                    service_container_logger.info(f"Shutting down service '{name}'...")
                    try:
                        shutdown_method = getattr(instance, shutdown_method_name)
                        if asyncio.iscoroutinefunction(shutdown_method):
                            await shutdown_method()
                        else:
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, shutdown_method)
                        self._service_states[name] = ServiceLifecycleState.SHUTDOWN
                        service_container_logger.info(
                            f"Service '{name}' shut down successfully."
                        )
                    except Exception as e:
                        service_container_logger.error(
                            f"Error shutting down service '{name}'.", exception=e
                        )
                else:
                    service_container_logger.debug(
                        f"Service '{name}' has no specific shutdown method."
                    )

        self._services.clear()
        self._service_factories.clear()
        self._service_states.clear()
        service_container_logger.info("All services shutdown process completed.")

    @detailed_log_function(LogCategory.SYSTEM)
    async def get_system_health_summary(self) -> Dict[str, Any]:
        """Aggregates health status from all registered, initialized services."""
        service_container_logger.info("Aggregating system health summary.")
        overall_healthy_count = 0
        total_checked_services = 0
        services_status_map: Dict[str, Any] = {}

        for name, instance in self._services.items():
            if self._service_states.get(name) == ServiceLifecycleState.INITIALIZED:
                total_checked_services += 1
                service_status = {
                    "status": "unknown",
                    "details": "No health check method.",
                }
                health_method_name = None
                if hasattr(instance, "get_service_status"):
                    health_method_name = "get_service_status"
                elif hasattr(instance, "health_check"):
                    health_method_name = "health_check"

                if health_method_name:
                    try:
                        health_method = getattr(instance, health_method_name)
                        if asyncio.iscoroutinefunction(health_method):
                            status_report = await health_method()
                        else:
                            status_report = health_method()

                        services_status_map[name] = status_report
                        if status_report.get("status") == "healthy":
                            overall_healthy_count += 1
                    except Exception as e:
                        service_container_logger.warning(
                            f"Health check failed for service '{name}'.", exception=e
                        )
                        services_status_map[name] = {
                            "status": "error",
                            "details": str(e),
                        }
                else:
                    services_status_map[name] = service_status  # No health check method
            elif self._service_states.get(name) == ServiceLifecycleState.ERROR:
                services_status_map[name] = {"status": "error_on_init"}

        overall_status_str = "healthy"
        if (
            total_checked_services == 0 and self._services
        ):  # Services registered but none initialized correctly
            overall_status_str = "error_no_services_initialized"
        elif overall_healthy_count < total_checked_services:
            overall_status_str = "degraded"
        elif not self._services:  # No services registered at all
            overall_status_str = "empty_no_services_registered"

        summary = {
            "overall_status": overall_status_str,
            "total_services_registered": len(self._initialization_order),
            "total_services_initialized_and_checked": total_checked_services,
            "healthy_services_count": overall_healthy_count,
            "services_status": services_status_map,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        service_container_logger.info(
            "System health summary generated.", parameters=summary
        )
        return summary

    def add_background_task(self, coro: Awaitable[Any]):
        """Adds an awaitable to be run as a background task, managed by the container."""
        task = asyncio.create_task(coro)
        self._async_tasks.append(task)
        service_container_logger.info(
            "Background task added to ServiceContainer.",
            parameters={"task_name": getattr(coro, "__name__", "unnamed_coro")},
        )


# Global factory function to create and populate the service container
# This is where you define how your system's services are created and wired together.
async def create_service_container(
    app_settings: Optional[Any] = None,
) -> ServiceContainer:  # app_settings can be LegalAISettings
    """
    Factory function to create and populate the ServiceContainer with all core services.
    This is the main dependency injection setup for the application.
    """
    service_container_logger.info("=== CREATE SERVICE CONTAINER START ===")
    container = ServiceContainer()

    # 1. Configuration Manager (must be first)
    from ..core.configuration_manager import create_configuration_manager

    # If app_settings (e.g. LegalAISettings from config.settings) is passed, use it
    await container.register_service(
        "configuration_manager",
        factory=lambda sc, custom_settings_instance=app_settings: create_configuration_manager(
            custom_settings_instance=custom_settings_instance
        ),
    )
    config_manager_service = await container.get_service("configuration_manager")

    # 2. Core Services (Loggers are implicitly available via get_detailed_logger)
    # ErrorHandler is a global singleton, usually not registered but can be if needed for explicit access.
    # from .unified_exceptions import get_error_handler
    # await container.register_service("error_handler", instance=get_error_handler())

    # 3. Persistence Layer (Moved up as UserRepository depends on it)
    from ..core.enhanced_persistence import (
        create_enhanced_persistence_manager,
    )

    db_conf = config_manager_service.get_database_config()
    persistence_cfg_for_factory = {
        "database_url": db_conf.get(
            "neo4j_uri"
        ),  # Example, if EnhancedPersistence uses Neo4j as primary
        # Or better: db_conf.get_url_for_service("main_relational_db")
        "redis_url": config_manager_service.get("REDIS_URL_CACHE"),  # Example
        "persistence_config": config_manager_service.get(
            "persistence_layer_details", {}
        ),
    }
    await container.register_service(
        "persistence_manager",
        factory=create_enhanced_persistence_manager,
        is_async_factory=False,
        config=persistence_cfg_for_factory,
    )

    # Get PersistenceManager first as UserRepository depends on its pool
    persistence_manager_service = await container.get_service("persistence_manager")
    if (
        not persistence_manager_service or not persistence_manager_service.initialized
    ):  # Check if it got initialized by get_service
        # If it has an initialize method, initialize_all_services should handle it.
        # For now, we assume get_service might initialize it or it's initialized if instance is directly registered.
        # Re-checking state after attempting to get it.
        if hasattr(persistence_manager_service, "initialize") and not getattr(
            persistence_manager_service, "initialized", True
        ):
            service_container_logger.info(
                "Explicitly initializing persistence_manager after fetching."
            )
            await persistence_manager_service.initialize()

    if not persistence_manager_service or not getattr(
        persistence_manager_service, "initialized", False
    ):  # Final check before use
        service_container_logger.critical(
            "PersistenceManager not available or not initialized. Cannot proceed with services depending on it."
        )
        # Handle this critical failure

    # Register UserRepository
    from ..utils.user_repository import UserRepository
    if persistence_manager_service and persistence_manager_service.connection_pool:
        await container.register_service(
            "user_repository",
            instance=UserRepository(persistence_manager_service.connection_pool),
        )
    else:
        service_container_logger.warning(
            "UserRepository not registered due to missing PersistenceManager or its connection pool."
        )

    # Modify SecurityManager registration to inject UserRepository
    from .security_manager import SecurityManager

    sec_config = config_manager_service.get_security_config()
    enc_pass = os.getenv(
        "LEGAL_AI_ENCRYPTION_PASSWORD_SECRET", "default_dev_password_CHANGE_ME_IN_PROD!"
    )
    user_repo_instance = (
        await container.get_service("user_repository")
        if "user_repository" in container._services
        else None
    )  # Get if registered

    # Encryption password should be from a secure source (env var, secret manager)
    await container.register_service(
        "security_manager",
        instance=SecurityManager(
            encryption_password=enc_pass,
            allowed_directories=sec_config.get("allowed_directories", []),
        ),
    )
    # SecurityManager's AuthManager needs the UserRepository
    # So, SecurityManager might need to get UserRepository from the container itself,
    # or AuthManager is created with UserRepository and then passed to SecurityManager.
    # Let's assume SecurityManager's __init__ is adapted to take UserRepository for its AuthManager.
    # For simplicity here, we'll assume SecurityManager's constructor can handle an optional user_repository.
    # A cleaner way is for SecurityManager to fetch UserRepository from the container in its own init.

    # Simpler: AuthManager is created with repo, then passed to SecurityManager
    auth_manager_instance = AuthenticationManager(user_repository=user_repo_instance)

    # SecurityManager now takes an AuthenticationManager instance
    # This requires changing SecurityManager's __init__ signature
    # Original: SecurityManager(encryption_password, allowed_directories)
    # New (example): SecurityManager(encryption_password, allowed_directories, auth_manager_instance)

    # For now, let's assume SecurityManager's __init__ will fetch UserRepository itself if needed,
    # or its AuthenticationManager is initialized with it.
    # The provided SecurityManager.__init__ does not take UserRepository.
    # So, either SecurityManager needs to be refactored to accept/fetch it for its AuthManager,
    # or AuthManager needs to be a separate service.
    # Let's make AuthManager a separate service for clarity.

    await container.register_service(
        "authentication_manager", instance=auth_manager_instance
    )

    # ... (rest of service registrations) ...

    # After core services have been registered continue with LLM setup
    from ..core.llm_providers import (
        LLMManager,
        LLMConfig,
        LLMProviderEnum,
    )  # Assuming in core

    llm_primary_conf_dict = config_manager_service.get_llm_config()
    primary_llm_config = LLMConfig(**llm_primary_conf_dict)  # Create Pydantic model
    # Example fallback config (could also come from ConfigurationManager)
    fallback_llm_conf_dict = config_manager_service.get(
        "llm_fallback_config",
        {  # Example key
            "provider": LLMProviderEnum.OLLAMA,
            "model": "llama3.1:8b-instruct-q4_K_M",
            "base_url": "http://localhost:11434",
        },
    )
    fallback_llm_config = LLMConfig(**fallback_llm_conf_dict)
    await container.register_service(
        "llm_manager",
        instance=LLMManager(
            primary_config=primary_llm_config, fallback_configs=[fallback_llm_config]
        ),
    )

    from ..core.model_switcher import ModelSwitcher

    llm_manager_service = await container.get_service("llm_manager")
    # ModelSwitcher needs API key if it's to create new configs for XAI/OpenAI
    # This key should come from a secure source, e.g. config_manager
    api_key_for_switcher = config_manager_service.get(
        f"{llm_primary_conf_dict['provider']}_api_key"
    )  # e.g. xai_api_key
    await container.register_service(
        "model_switcher",
        instance=ModelSwitcher(
            llm_manager_service, default_provider_type=primary_llm_config.provider
        ),
    )

    from ..core.embedding_manager import EmbeddingManager

    embed_conf = (
        config_manager_service.get_vector_store_config()
    )  # Embedding model is part of VS config
    await container.register_service(
        "embedding_manager",
        instance=EmbeddingManager(
            model_name=embed_conf.get("embedding_model", "all-MiniLM-L6-v2"),
            cache_dir_str=str(
                config_manager_service.get("data_dir") / "cache" / "embeddings"
            ),
        ),
    )

    # 4. Knowledge Layer (Persistence Layer was moved up)
    from .knowledge_graph_manager import create_knowledge_graph_manager

    kg_conf = {  # Fetch from config_manager
        "NEO4J_URI": db_conf.get("neo4j_uri"),
        "NEO4J_USER": db_conf.get("neo4j_user"),
        "NEO4J_PASSWORD": db_conf.get("neo4j_password"),
        "ENABLE_NEO4J_PERSISTENCE": True,
    }
    await container.register_service(
        "knowledge_graph_manager",
        factory=create_knowledge_graph_manager,
        service_config=kg_conf,
    )

    from ..core.vector_store import create_vector_store  # Standard one

    vs_conf = {  # Fetch from config_manager
        "STORAGE_PATH": str(
            config_manager_service.get("data_dir") / "vector_store_main"
        ),
        "DEFAULT_INDEX_TYPE": embed_conf.get(
            "vector_store_type", "HNSW"
        ),  # Map if needed
        "embedding_model_name": embed_conf.get("embedding_model"),
    }
    # EmbeddingProvider instance can be fetched from EmbeddingManager if VectorStore is designed to take it
    # embedding_provider_instance = await container.get_service("embedding_manager").get_provider_instance() # Conceptual
    await container.register_service(
        "vector_store", factory=create_vector_store, service_config=vs_conf
    )

    from ..core.optimized_vector_store import (
        create_optimized_vector_store,
    )  # Optimized one

    ovs_conf = {  # Can have its own config or inherit
        "STORAGE_PATH": str(
            config_manager_service.get("data_dir") / "vector_store_optimized"
        ),
        "DEFAULT_INDEX_TYPE": "HNSW",  # Often optimized means HNSW or specific FAISS params
    }
    await container.register_service(
        "optimized_vector_store",
        factory=create_optimized_vector_store,
        service_config=ovs_conf,
    )

    from .realtime_graph_manager import create_realtime_graph_manager

    kgm_service = await container.get_service("knowledge_graph_manager")
    ovs_service = await container.get_service("optimized_vector_store")
    await container.register_service(
        "realtime_graph_manager",
        factory=create_realtime_graph_manager,
        kg_manager=kgm_service,
        vector_store=ovs_service,
    )

    # 5. Memory Layer
    from ..core.unified_memory_manager import create_unified_memory_manager

    umm_conf = {"DB_PATH": str(db_conf.get("memory_db_path"))}
    await container.register_service(
        "unified_memory_manager",
        factory=create_unified_memory_manager,
        service_config=umm_conf,
    )

    from ..utils.reviewable_memory import create_reviewable_memory

    umm_service = await container.get_service("unified_memory_manager")
    revmem_conf = {
        "DB_PATH": str(
            config_manager_service.get("data_dir") / "databases" / "review_memory.db"
        )
    }
    await container.register_service(
        "reviewable_memory",
        factory=create_reviewable_memory,
        service_config=revmem_conf,
        unified_memory_manager=umm_service,
    )

    from .violation_review import ViolationReviewDB

    violation_db_path = str(db_conf.get("violations_db_path"))
    await container.register_service(
        "violation_review_db",
        instance=ViolationReviewDB(db_path=violation_db_path),
    )

    from .workflow_orchestrator import WorkflowOrchestrator
    await container.register_service(
        "workflow_orchestrator",
        factory=lambda sc: WorkflowOrchestrator(sc),
    )

    # 6. Agents (Register factories for agents)
    # Agents are often stateful per task, so factories are common.
    # Or, if stateless, can be singletons. BaseAgent is typically instantiated per use or task.
    # Here we register the classes themselves, and workflows will instantiate them.
    # If agents are true "services" (long-lived, shared), then register instances.
    # For now, let's assume workflows will get agent *classes* or factories.
    # Or, if agents are simple enough to be singletons:
    from ..agents.document_processor_agent import DocumentProcessorAgent
    from ..agents.document_rewriter_agent import DocumentRewriterAgent
    from ..agents.ontology_extraction_agent import OntologyExtractionAgent
    from ..agents.entity_extraction_agent import StreamlinedEntityExtractionAgent

    # Example: await container.register_service("document_processor_agent", instance=DocumentProcessorAgent(container))
    # This needs careful thought: are agents services or instantiated by workflows?
    # The original code often had agents take 'services' in __init__, implying they are created with access to container.
    # Let's register them as factories that take the container.

    agent_classes = {
        "document_processor_agent": DocumentProcessorAgent,
        "ontology_extraction_agent": OntologyExtractionAgent,
        "streamlined_entity_extraction_agent": StreamlinedEntityExtractionAgent,
        # ... Add all other agent classes from agents/__init__.py
        "semantic_analysis_agent": getattr(
            __import__("legal_ai_system.agents", fromlist=["SemanticAnalysisAgent"]),
            "SemanticAnalysisAgent",
            None,
        ),
        "structural_analysis_agent": getattr(
            __import__("legal_ai_system.agents", fromlist=["StructuralAnalysisAgent"]),
            "StructuralAnalysisAgent",
            None,
        ),
        "citation_analysis_agent": getattr(
            __import__("legal_ai_system.agents", fromlist=["CitationAnalysisAgent"]),
            "CitationAnalysisAgent",
            None,
        ),
        "text_correction_agent": getattr(
            __import__("legal_ai_system.agents", fromlist=["TextCorrectionAgent"]),
            "TextCorrectionAgent",
            None,
        ),
        "document_rewriter_agent": DocumentRewriterAgent,
        "violation_detector_agent": getattr(
            __import__("legal_ai_system.agents", fromlist=["ViolationDetectorAgent"]),
            "ViolationDetectorAgent",
            None,
        ),
        "auto_tagging_agent": getattr(
            __import__("legal_ai_system.agents", fromlist=["AutoTaggingAgent"]),
            "AutoTaggingAgent",
            None,
        ),
        "note_taking_agent": getattr(
            __import__("legal_ai_system.agents", fromlist=["NoteTakingAgent"]),
            "NoteTakingAgent",
            None,
        ),
        "legal_analysis_agent": getattr(
            __import__("legal_ai_system.agents", fromlist=["LegalAnalysisAgent"]),
            "LegalAnalysisAgent",
            None,
        ),
        "knowledge_base_agent": getattr(
            __import__("legal_ai_system.agents", fromlist=["KnowledgeBaseAgent"]),
            "KnowledgeBaseAgent",
            None,
        ),
    }
    for name, agent_cls in agent_classes.items():
        if agent_cls:  # Check if import was successful
            agent_config = config_manager_service.get(
                f"agents.{name}_config", {}
            )  # Agent specific config
            # Pass the service_container (self) to the factory
            await container.register_service(
                name,
                factory=lambda sc, cfg=agent_config, cls=agent_cls: cls(sc, **cfg),
                is_async_factory=False,
            )

    # Initialize all services that were registered with factories or need explicit init
    await container.initialize_all_services()

    service_container_logger.info("=== CREATE SERVICE CONTAINER END ===")
    return container
