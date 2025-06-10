"""Public re-export of project settings."""
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
