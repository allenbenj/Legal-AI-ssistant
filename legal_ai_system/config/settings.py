"""Re-export configuration settings for external use.

This module exposes the main :class:`LegalAISettings` model along with helper
functions used across the application. Importing from this module avoids
having to know the internal :mod:`legal_ai_system.core` package structure.
"""

from legal_ai_system.core.settings import (
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
