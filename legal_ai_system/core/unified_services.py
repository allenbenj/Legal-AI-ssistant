"""Simplified service registry used for tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from ..services.violation_review import ViolationReviewManager
from .vector_store_manager import VectorStoreManager


class UnifiedMemoryManager:
    """Lightweight stand-in for the real memory manager."""

    def __init__(self, db_path: str = "./storage/databases/unified_memory.db") -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        """Placeholder initializer for compatibility."""
        return None


class ServicePriority(Enum):
    """Relative priority for registered services."""

    LOW = auto()
    NORMAL = auto()
    HIGH = auto()

@dataclass
class ServiceMetadata:
    """Metadata describing a service instance."""

    instance: Any
    priority: ServicePriority = ServicePriority.NORMAL
    dependencies: List[str] = field(default_factory=list)

class ServiceContainer:
    """Basic container storing service instances and metadata."""

    def __init__(self) -> None:
        self._services: Dict[str, ServiceMetadata] = {}

    def register_service(
        self,
        name: str,
        instance: Any,
        priority: ServicePriority = ServicePriority.NORMAL,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Register a new service instance."""
        self._services[name] = ServiceMetadata(
            instance=instance,
            priority=priority,
            dependencies=dependencies or [],
        )

    def get_service(self, name: str) -> Any:
        """Retrieve a previously registered service."""
        return self._services[name].instance

_container: Optional[ServiceContainer] = None

def get_service_container() -> ServiceContainer:
    """Return the global service container, creating it if necessary."""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container

def register_core_services() -> None:
    """Register essential backend services."""
    container = get_service_container()
    if "vector_store_manager" not in container._services:
        container.register_service(
            "vector_store_manager",
            VectorStoreManager(),
            priority=ServicePriority.NORMAL,
            dependencies=[],
        )
    if "unified_memory_manager" not in container._services:
        container.register_service(
            "unified_memory_manager",
            UnifiedMemoryManager(),
            priority=ServicePriority.NORMAL,
            dependencies=[],
        )
    if "violation_review_manager" not in container._services:
        container.register_service(
            "violation_review_manager",
            ViolationReviewManager(),
            priority=ServicePriority.NORMAL,
            dependencies=[],
        )

