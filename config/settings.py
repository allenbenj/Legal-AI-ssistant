"""Public re-export of project settings."""

from legal_ai_system.config import settings as _core_settings

# Re-export selected attributes while keeping linters happy by assigning them.
LegalAISettings = _core_settings.LegalAISettings
get_db_url = _core_settings.get_db_url
get_vector_store_path = _core_settings.get_vector_store_path
is_supported_file = _core_settings.is_supported_file
settings = _core_settings.settings

__all__ = [
    "LegalAISettings",
    "settings",
    "get_db_url",
    "get_vector_store_path",
    "is_supported_file",
]
