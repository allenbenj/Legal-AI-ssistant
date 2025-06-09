from typing import Optional

from ..services.service_container import ServiceContainer




def get_service_container() -> ServiceContainer:
    """Return the singleton service container, creating it if needed."""
    if _service_container is None:
