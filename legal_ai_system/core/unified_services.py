from __future__ import annotations

"""Simplified service registration used in tests.

This module exposes ``get_service_container`` and ``register_core_services`` as
lightweight wrappers around :class:`ServiceContainer`. It avoids pulling in the
full application configuration while satisfying the minimal expectations of the
unit tests.
"""

import asyncio
from typing import Optional

from ..services.service_container import ServiceContainer

# Singleton container used by tests
_service_container: Optional[ServiceContainer] = None

async def _basic_registration(container: ServiceContainer) -> None:
    """Register minimal services required for the tests."""
    # The real project would wire concrete implementations here. For the test
    # suite we just register simple placeholder objects.
    await container.register_service("vector_store_manager", instance=object())
    await container.register_service("unified_memory_manager", instance=object())


def register_core_services() -> ServiceContainer:
    """Create the global service container and register core services."""
    global _service_container
    if _service_container is None:
        _service_container = ServiceContainer()
        asyncio.run(_basic_registration(_service_container))
    return _service_container


def get_service_container() -> ServiceContainer:
    """Return the singleton service container, creating it if needed."""
    if _service_container is None:
        return register_core_services()
    return _service_container

