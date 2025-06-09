"""Minimal unified services module used in tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..services.service_container import ServiceContainer

__all__ = ["ServiceMeta", "register_core_services", "get_service_container"]


@dataclass
class ServiceMeta:
    priority: type
    dependencies: List[str]


_container = ServiceContainer()


def register_core_services() -> None:
    """Register placeholder services used in tests."""

    class _Priority:
        name = "HIGH"

    _container._services["vector_store_manager"] = ServiceMeta(_Priority, [])
    _container._services["unified_memory_manager"] = ServiceMeta(_Priority, [])


def get_service_container() -> ServiceContainer:
    return _container
