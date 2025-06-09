"""Utility helpers to provide a global service container."""
from __future__ import annotations

import asyncio
from typing import Optional

from ..services.service_container import ServiceContainer
from .vector_store_manager import create_vector_store_manager
from .unified_memory_manager import create_unified_memory_manager

_container: Optional[ServiceContainer] = None


def get_service_container() -> ServiceContainer:
    """Return a singleton :class:`ServiceContainer`."""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def register_core_services() -> None:
    """Register key core services in the container."""

    async def _register(container: ServiceContainer) -> None:
        if "vector_store_manager" not in container._services:
            await container.register_service(
                "vector_store_manager", instance=create_vector_store_manager()
            )
        if "unified_memory_manager" not in container._services:
            await container.register_service(
                "unified_memory_manager", instance=create_unified_memory_manager()
            )

    container = get_service_container()
    asyncio.run(_register(container))


def shutdown_core_services() -> None:
    """Shutdown and reset the global service container."""
    container = get_service_container()
    asyncio.run(container.shutdown_all_services())
