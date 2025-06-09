"""Public re-export of project settings."""

from legal_ai_system.config import settings as _core_settings

# Mirror the re-exports from ``legal_ai_system.config.settings`` while
# keeping the imported names referenced so IDEs don't mark them as
# unused imports.
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
