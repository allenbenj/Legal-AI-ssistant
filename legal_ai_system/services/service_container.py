# legal_ai_system/core/service_container.py
"""
Service Container for Dependency Injection and Lifecycle Management.

Manages the creation, retrieval, initialization, and shutdown of all
core services and agents within the Legal AI System.
"""

import asyncio
from typing import (
    Dict,
    Any,
    Optional,
    Callable,
    Awaitable,
    List,
    TYPE_CHECKING,
    TypeVar,
)
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

    try:
        from .realtime_analysis_workflow import RealTimeAnalysisWorkflow
    except Exception:  # pragma: no cover - optional during tests
        RealTimeAnalysisWorkflow = None  # type: ignore
    from .workflow_config import WorkflowConfig


# Initialize logger for this module
service_container_logger: DetailedLogger = get_detailed_logger(
    "ServiceContainer", LogCategory.SYSTEM
)

T = TypeVar("T")


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
        # Active workflow configuration shared across workflow instances
        self._active_workflow_config: Dict[str, Any] = {}

        service_container_logger.info("ServiceContainer instance created.")

    def update_workflow_config(self, config: Dict[str, Any]) -> None:
        """Update the active workflow configuration in-place."""
        self._active_workflow_config.update(config)

    def get_active_workflow_config(self) -> Dict[str, Any]:
        """Return a copy of the active workflow configuration."""
        return dict(self._active_workflow_config)

    @detailed_log_function(LogCategory.SYSTEM)
    async def register_service(
        self,
        name: str,
        instance: Optional[T] = None,
        factory: Optional[Callable[..., Awaitable[T] | T]] = None,
        is_async_factory: bool = False,
        depends_on: Optional[List[str]] = None,
        config_key: Optional[str] = None,  # Key to fetch config for this service
        **factory_kwargs: Any,
    ) -> None:
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
    async def get_service(self, name: str, _chain: Optional[List[str]] = None) -> T:
        """Retrieves a service instance, creating it via factory if necessary."""
        if _chain is None:
            _chain = []

        if name in _chain:
            chain = " -> ".join(_chain + [name])
            service_container_logger.critical(
                f"Circular dependency detected while resolving {chain}."
            )
            raise ConfigurationError(f"Circular dependency detected: {chain}")

        async with self._lock:
            if name in self._services:
                return self._services[name]  # type: ignore[return-value]

            if name not in self._service_factories:
                service_container_logger.error(
                    f"Service not found.", parameters={"name": name}
                )
                raise ConfigurationError(f"Service '{name}' not found in container.")

            # Copy factory info while holding the lock then release
            factory_info = self._service_factories[name]

        service_container_logger.info(f"Creating service '{name}' from factory.")

        # Resolve dependencies first outside the lock
        for dep_name in factory_info["depends_on"]:
            if dep_name not in self._services:
                await self.get_service(dep_name, _chain + [name])

        # Get config for the service if config_key is provided
        service_config = {}
        if factory_info["config_key"]:
            config_manager = self._services.get(
                "configuration_manager"
            )  # Assume CM is registered
            if config_manager and hasattr(config_manager, "get"):
                service_config = config_manager.get(factory_info["config_key"], {})
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

            async with self._lock:
                self._services[name] = instance
            return instance
        except Exception as e:
            service_container_logger.error(
                f"Failed to create service '{name}'.", exception=e
            )
            raise

    @detailed_log_function(LogCategory.SYSTEM)
    async def initialize_all_services(self) -> None:
        """Initializes all registered services that have an 'initialize_service' or 'initialize' method."""
        service_container_logger.info("Initializing all registered services...")
        # Sort services by dependency order if complex dependencies exist (topological sort)
        # For now, using registration order.

        # First, instantiate all factory-based services that haven't been created yet
        for name in list(
            self._service_factories.keys()
        ):  # Iterate on copy as get_service modifies _services
            if name not in self._services:
                try:
                    service_container_logger.debug(
                        f"Creating service '{name}' before initialization."
                    )
                    await self.get_service(name)  # This will create it
                except Exception as e:
                    service_container_logger.error(
                        f"Failed to create service '{name}' prior to initialization.",
                        exception=e,
                    )
                    continue

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
    async def shutdown_all_services(self) -> None:
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

    @detailed_log_function(LogCategory.SYSTEM)
    def get_services_status(self) -> Dict[str, Any]:
        """Return lifecycle state and configuration of registered services."""
        services_data: Dict[str, Any] = {}
        for name in self._initialization_order:
            state = self._service_states.get(name, ServiceLifecycleState.REGISTERED)
            factory_info = self._service_factories.get(name, {})
            services_data[name] = {
                "state": state.value,
                "config_key": factory_info.get("config_key"),
                "factory_kwargs": {k: repr(v) for k, v in factory_info.get("kwargs", {}).items()},
            }

        overview = {
            "services": services_data,
            "active_workflow_config": self.get_active_workflow_config(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        service_container_logger.info(
            "Services status overview generated.", parameters=overview
        )
        return overview

    def add_background_task(self, coro: Awaitable[Any]) -> None:
        """Adds an awaitable to be run as a background task, managed by the container."""
        task = asyncio.create_task(coro)
        self._async_tasks.append(task)
        service_container_logger.info(
            "Background task added to ServiceContainer.",
            parameters={"task_name": getattr(coro, "__name__", "unnamed_coro")},
        )

    async def update_workflow_config(self, new_config: Dict[str, Any]) -> None:
        """Merge new values into the active workflow configuration."""
        async with self._lock:
            self._active_workflow_config.update(new_config)
            service_container_logger.info(
                "Workflow configuration updated.",
                parameters={"config": self._active_workflow_config},
            )

    def get_active_workflow_config(self) -> Dict[str, Any]:
        """Return a copy of the currently active workflow configuration."""
        return dict(self._active_workflow_config)


# ----- Service Factories -------------------------------------------------
def create_connection_pool(
    service_container: "ServiceContainer",
    database_url: Optional[str] = None,
    redis_url: Optional[str] = None,
    min_pg_connections: int = 5,
    max_pg_connections: int = 20,
    max_redis_connections: int = 10,
) -> "ConnectionPool":
    """Factory for :class:`ConnectionPool` used by the persistence layer."""
    from ..core.enhanced_persistence import ConnectionPool

    return ConnectionPool(
        database_url,
        redis_url,
        min_pg_connections,
        max_pg_connections,
        max_redis_connections,
    )


def create_persistence_manager(
    service_container: "ServiceContainer",
    connection_pool: "ConnectionPool",
    config: Optional[Dict[str, Any]] = None,
    metrics_exporter: Optional[Any] = None,
) -> Any:
    """Factory for the persistence manager."""
    from ..core.enhanced_persistence import EnhancedPersistenceManager

    return EnhancedPersistenceManager(
        connection_pool=connection_pool,
        config=config or {},
        metrics_exporter=metrics_exporter,
    )


async def register_all_agents(
    container: "ServiceContainer", config_manager_service: Any
) -> None:
    """Discover agents under ``legal_ai_system/agents`` and register them.

    Modules are not imported until the agent instance is requested to avoid
    heavy imports during container setup.  Each agent is registered under a
    lowercase key derived from the class name.
    """
    import ast
    from importlib import import_module
    from pathlib import Path

    agents_path = Path(__file__).resolve().parents[1] / "agents"
    for file in agents_path.glob("*.py"):
        module_name = f"legal_ai_system.agents.{file.stem}"
        with open(file, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=str(file))

        class_names = [
            n.name
            for n in tree.body
            if isinstance(n, ast.ClassDef)
            and (n.name.endswith("Agent") or n.name.endswith("Engine"))
        ]
        for cls_name in class_names:
            agent_id = cls_name.lower()
            agent_config = config_manager_service.get(
                f"agents.{agent_id}_config", {}
            )

            async def factory(sc, *, m=module_name, c=cls_name, cfg=agent_config):
                mod = import_module(m)
                cls = getattr(mod, c)
                return cls(sc, **cfg)

            await container.register_service(
                agent_id, factory=factory, is_async_factory=True
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

    # Register the metrics exporter after config so it can read settings if needed
    from .metrics_exporter import init_metrics_exporter

    await container.register_service(
        "metrics_exporter",
        instance=init_metrics_exporter(),
    )

    # 2. Database Connection Pool
    db_conf = config_manager_service.get_database_config()
    await container.register_service(
        "connection_pool",
        factory=lambda sc, db_url=db_conf.database_url, redis_url=db_conf.redis_url_cache: create_connection_pool(
            sc,
            database_url=db_url,
            redis_url=redis_url,
        ),
        is_async_factory=False,
    )

    connection_pool_service = await container.get_service("connection_pool")

    from .database_manager import DatabaseManager
    db_path = str(config_manager_service.get("data_dir") / "databases" / "legal_ai_gui.db")
    await container.register_service(
        "database_manager",
        instance=DatabaseManager(db_path),
    )

    # 3. Persistence Manager
    persistence_cfg = config_manager_service.get("persistence_layer_details", {})
    await container.register_service(
        "persistence_manager",
        factory=create_persistence_manager,
        is_async_factory=False,
        connection_pool=connection_pool_service,
        config=persistence_cfg,
        metrics_exporter=await container.get_service("metrics_exporter"),
    )

    # Task Queue setup for background processing
    from .task_queue import TaskQueue

    queue_url = db_conf.redis_url_queue or config_manager_service.get(
        "REDIS_URL_QUEUE", "redis://localhost:6379/0"
    )
    await container.register_service(
        "task_queue",
        instance=TaskQueue(redis_url=queue_url),
    )

    # 2. Core Services (Loggers are implicitly available via get_detailed_logger)
    # ErrorHandler is a global singleton, usually not registered but can be if needed for explicit access.
    # from .unified_exceptions import get_error_handler
    # await container.register_service("error_handler", instance=get_error_handler())

    # 3. Persistence Layer (moved above)

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

    if connection_pool_service:
        await container.register_service(
            "user_repository",
            instance=UserRepository(connection_pool_service),
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
            allowed_directories=sec_config.allowed_directories,
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

    llm_primary_conf = config_manager_service.get_llm_config()
    primary_llm_config = llm_primary_conf
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
        f"{llm_primary_conf.provider}_api_key"
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
            model_name=embed_conf.embedding_model,
            cache_dir_str=str(
                config_manager_service.get("data_dir") / "cache" / "embeddings"
            ),
        ),
    )

    # 4. Knowledge Layer (Persistence Layer was moved up)
    from .knowledge_graph_manager import create_knowledge_graph_manager

    kg_conf = {  # Fetch from config_manager
        "NEO4J_URI": db_conf.neo4j_uri,
        "NEO4J_USER": db_conf.neo4j_user,
        "NEO4J_PASSWORD": db_conf.neo4j_password,
        "ENABLE_NEO4J_PERSISTENCE": True,
    }
    await container.register_service(
        "knowledge_graph_manager",
        factory=create_knowledge_graph_manager,
        connection_pool=connection_pool_service,
        config=kg_conf,
    )

    from ..core.enhanced_vector_store import (
        create_enhanced_vector_store,
    )  # Unified implementation

    vs_conf = {
        "STORAGE_PATH": str(
            config_manager_service.get("data_dir") / "vector_store_main"
        ),
        "DOCUMENT_INDEX_PATH": str(embed_conf.document_index_path),
        "ENTITY_INDEX_PATH": str(embed_conf.entity_index_path),
        "DEFAULT_INDEX_TYPE": embed_conf.type,
        "embedding_model_name": embed_conf.embedding_model,
    }
    # EmbeddingProvider instance can be fetched from EmbeddingManager if VectorStore is designed to take it
    # embedding_provider_instance = await container.get_service("embedding_manager").get_provider_instance() # Conceptual
    await container.register_service(
        "vector_store",
        factory=create_enhanced_vector_store,
        connection_pool=connection_pool_service,
        config=vs_conf,
    )

    from .realtime_graph_manager import create_realtime_graph_manager

    kgm_service = await container.get_service("knowledge_graph_manager")
    vs_service = await container.get_service("vector_store")
    await container.register_service(
        "realtime_graph_manager",
        factory=create_realtime_graph_manager,
        kg_manager=kgm_service,
        vector_store=vs_service,
    )

    # 5. Memory Layer
    from ..core.unified_memory_manager import (
        create_unified_memory_manager,
        UnifiedMemoryManagerConfig,
    )

    umm_conf = UnifiedMemoryManagerConfig(db_path=db_conf.memory_db_path)
    await container.register_service(
        "unified_memory_manager",
        factory=create_unified_memory_manager,
        service_config=umm_conf,
    )

    from ..core.ml_optimizer import create_ml_optimizer

    await container.register_service(
        "ml_optimizer",
        factory=create_ml_optimizer,
    )
    ml_opt_service = await container.get_service("ml_optimizer")

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
        ml_optimizer=ml_opt_service,
    )

    from .violation_review import ViolationReviewDB

    violation_db_path = str(db_conf.violations_db_path)
    await container.register_service(
        "violation_review_db",
        instance=ViolationReviewDB(db_path=violation_db_path),
    )

    from .violation_classifier import ViolationClassifier

    vr_db_service = await container.get_service("violation_review_db")
    classifier = ViolationClassifier()
    classifier.train_from_review_db(vr_db_service)
    await container.register_service(
        "violation_classifier", instance=classifier
    )

    # Lightweight analytics utilities
    from .keyword_extraction_service import KeywordExtractionService
    from .quality_assessment_service import QualityAssessmentService

    await container.register_service(
        "keyword_extraction_service", instance=KeywordExtractionService()
    )
    await container.register_service(
        "quality_assessment_service", instance=QualityAssessmentService()
    )

    from .workflow_config import WorkflowConfig
    from .realtime_analysis_workflow import RealTimeAnalysisWorkflow

    workflow_cfg_dict = config_manager_service.get("workflow_config", {})
    container.update_workflow_config(workflow_cfg_dict)
    await container.register_service(
        "realtime_analysis_workflow",
        factory=lambda sc: RealTimeAnalysisWorkflow(
            sc,
            workflow_config=WorkflowConfig(**sc.get_active_workflow_config()),
            task_queue=sc._services.get("task_queue"),
        ),
        is_async_factory=False,
    )

    # Dynamically register all available agents. This defers module imports
    # until the agent is actually requested from the container.
    await register_all_agents(container, config_manager_service)

    # Register simple LangGraph node classes for builder workflows
    from ..agents.agent_nodes import AnalysisNode, SummaryNode

    workflow_topic = config_manager_service.get("workflow_builder_topic", "default")
    await container.register_service(
        "analysis_node",
        factory=lambda sc, t=workflow_topic: AnalysisNode(t),
        is_async_factory=False,
    )
    await container.register_service(
        "summary_node",
        factory=lambda sc: SummaryNode(),
        is_async_factory=False,
    )

    # Register LangGraph nodes and builder for the orchestrator
    from ..agents.agent_nodes import AnalysisNode, SummaryNode
    from ..workflows.langgraph_setup import build_graph
    from .workflow_orchestrator import WorkflowOrchestrator

    await container.register_service(
        "analysis_node_factory",
        factory=lambda sc, topic="default": AnalysisNode(topic),
        is_async_factory=False,
    )
    await container.register_service(
        "summary_node_factory",
        factory=lambda sc: SummaryNode(),
        is_async_factory=False,
    )
    await container.register_service(
        "langgraph_builder",
        instance=build_graph,
    )
    await container.register_service(
        "workflow_orchestrator",
        factory=lambda sc: WorkflowOrchestrator(
            sc,
            topic=workflow_topic,
            workflow_config=WorkflowConfig(**sc.get_active_workflow_config()),
        ),
        is_async_factory=False,
    )

    # Initialize all services that were registered with factories or need explicit init
    await container.initialize_all_services()

    service_container_logger.info("=== CREATE SERVICE CONTAINER END ===")
    return container
