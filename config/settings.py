"""Public re-export of project settings.

This module exposes the main ``LegalAISettings`` model and helper functions
from the package's internal configuration module.  It provides a stable import
path for consumers without requiring knowledge of the internal project
structure.
"""

from legal_ai_system.config.settings import (
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
