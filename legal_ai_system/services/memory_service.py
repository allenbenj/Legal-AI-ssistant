from __future__ import annotations

"""Utility context manager for accessing :class:`UnifiedMemoryManager`."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from ..core.unified_services import get_service_container, register_core_services
from ..core.unified_memory_manager import UnifiedMemoryManager


@asynccontextmanager
async def memory_manager_context() -> AsyncGenerator[UnifiedMemoryManager, None]:
    """Yield an initialized :class:`UnifiedMemoryManager` instance."""
    register_core_services()
    container = get_service_container()
    manager: UnifiedMemoryManager = await container.get_service("unified_memory_manager")
    if getattr(manager, "_initialized", False) is False:
        await manager.initialize()
    try:
        yield manager
    finally:
        # No explicit cleanup required
        pass
