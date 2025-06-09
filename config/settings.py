"""Public re-export of project settings."""

from legal_ai_system.config import settings as _core_settings


__all__ = [
    "LegalAISettings",
    "settings",
    "get_db_url",
    "get_vector_store_path",
    "is_supported_file",
]
