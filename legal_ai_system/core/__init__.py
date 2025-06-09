"""Core package utilities and service access points."""

from .unified_services import get_service_container, register_core_services

__all__ = [
    "get_service_container",
    "register_core_services",
]

