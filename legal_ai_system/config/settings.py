"""Re-export configuration settings for external use.

This module exposes the :class:`LegalAISettings` model and related helpers from
``legal_ai_system.core.settings`` so that callers can simply import from the
``config`` package without being coupled to the internal directory layout.
"""

from ..core.settings import (
    LegalAISettings,
    settings,
    get_db_url,
    get_vector_store_path,
    is_supported_file,
)

__all__ = [
    "LegalAISettings",
    "settings",
    "get_db_url",
    "get_vector_store_path",
    "is_supported_file",
]
