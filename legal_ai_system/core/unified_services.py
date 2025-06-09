"""Minimal service registry used for tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ServiceMeta:
    factory: Any
    priority: Any = None
    dependencies: Any = None


class _Container:
    def __init__(self) -> None:
        self._services: Dict[str, ServiceMeta] = {}

    def register_service(self, name: str, factory: Any) -> None:
        self._services[name] = ServiceMeta(factory)


_container = _Container()


def register_core_services() -> None:
    """Register a minimal set of services for unit tests."""
    _container.register_service("vector_store_manager", object)
    _container.register_service("unified_memory_manager", object)


def get_service_container() -> _Container:
    return _container

__all__ = ["get_service_container", "register_core_services"]
