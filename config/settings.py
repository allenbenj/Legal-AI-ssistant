"""Public re-export of project settings."""

from legal_ai_system.config import settings as _settings


# Re-export specific names to keep this module as a lightweight facade
LegalAISettings = _settings.LegalAISettings
settings = _settings.settings
get_db_url = _settings.get_db_url
get_vector_store_path = _settings.get_vector_store_path
is_supported_file = _settings.is_supported_file

__all__ = [
    "LegalAISettings",
    "settings",
    "get_db_url",
    "get_vector_store_path",
    "is_supported_file",
]
