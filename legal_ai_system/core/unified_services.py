from typing import Optional

from ..services.service_container import ServiceContainer

_service_container: Optional[ServiceContainer] = None


def register_core_services() -> None:
    """Initialize a minimal service container for tests."""
    global _service_container
    if _service_container is None:
        container = ServiceContainer()
        # Register minimal placeholder services expected by tests
        container._services["vector_store_manager"] = object()
        container._services["unified_memory_manager"] = object()
        _service_container = container


def get_service_container() -> ServiceContainer:
    """Return the singleton service container, creating it if needed."""
    if _service_container is None:
        register_core_services()
    return _service_container
