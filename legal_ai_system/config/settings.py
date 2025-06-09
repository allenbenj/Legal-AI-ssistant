"""Re-export configuration settings for external use.

This module exposes the main :class:`LegalAISettings` model along with helper
functions used across the application.  Importing from this module avoids
having to know the internal ``core`` package structure.
"""

from ..core import settings as _core_settings

# Re-export individual attributes while keeping the import "used" so
# linters like PyDev don't report them as unused.  This pattern avoids
# the "unused import" warnings that can appear when names are only
# referenced in ``__all__``.
LegalAISettings = _core_settings.LegalAISettings
settings = _core_settings.settings
get_db_url = _core_settings.get_db_url
get_vector_store_path = _core_settings.get_vector_store_path
is_supported_file = _core_settings.is_supported_file

__all__ = [
    "LegalAISettings",
    "settings",
    "get_db_url",
    "get_vector_store_path",
    "is_supported_file",
]
