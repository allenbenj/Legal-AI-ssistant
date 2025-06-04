# legal_ai_system/core/configuration_manager.py
"""
Configuration Manager - Centralized Configuration Management
===========================================================
Provides a service-oriented interface to the Legal AI System configuration.
"""

from typing import Any, Dict, Optional, List, Union
from pathlib import Path
import os
import json # For serializing complex objects in logs if needed
import sys

# Import detailed logging system
from .detailed_logging import get_detailed_logger, LogCategory, detailed_log_function

# Import configuration settings
try:
    # This assumes settings.py is in ../config/ relative to this file (core/)
    from ..config.settings import settings as global_settings, LegalAISettings, get_db_url, get_vector_store_path, is_supported_file
except ImportError:
    # Fallback for scenarios where the relative import might fail (e.g., direct script execution or tests)
    # This might happen if the PYTHONPATH isn't set up as expected for the new structure yet.
    # For robustness in a large system, consider making imports absolute from the project root.
    # e.g., from legal_ai_system.config.settings import ...
    # For now, let's assume the relative import works in the final structure.
    # If `settings` is an object, `global_settings` is a good alias.
    # If `settings` is a module, then `from ..config import settings as global_settings_module` might be needed.
    # Given the original code, `settings` is an instance of `LegalAISettings`.
    class LegalAISettings: # Dummy for fallback
        def __init__(self, **data): pass
        def __getattr__(self, name): return None 
    
    global_settings = LegalAISettings() # Dummy instance
    
    def get_db_url(db_type: str) -> str: return ""
    def get_vector_store_path(store_type: str) -> Path: return Path(".")
    def is_supported_file(file_path: Union[str,Path]) -> bool: return False
    
    print("WARNING: ConfigurationManager using fallback imports for settings. Ensure correct project structure.", file=sys.stderr)


# Initialize logger
config_manager_logger = get_detailed_logger("ConfigurationManager", LogCategory.CONFIG) # Changed logger name


class ConfigurationManager:
    """
    Service-oriented configuration manager that provides centralized
    access to all Legal AI System configuration settings.
    """
    
    @detailed_log_function(LogCategory.CONFIG)
    def __init__(self, custom_settings_instance: Optional[LegalAISettings] = None): # Renamed for clarity
        """Initialize configuration manager with optional custom settings instance"""
        config_manager_logger.info("=== INITIALIZING CONFIGURATION MANAGER ===")
        
        self._settings: LegalAISettings = custom_settings_instance or global_settings
        self._config_cache: Dict[str, Any] = {} # For potentially computed or frequently accessed transformed configs
        self._environment_overrides: Dict[str, Any] = {}
        
        self._load_environment_overrides()
        
        config_manager_logger.info("ConfigurationManager initialized", parameters={
            'app_name': self._settings.app_name,
            'version': self._settings.version,
            'debug_mode': self._settings.debug,
            'llm_provider': self._settings.llm_provider,
            'vector_store_type': self._settings.vector_store_type,
            'environment_overrides_count': len(self._environment_overrides)
        })
    
    @detailed_log_function(LogCategory.CONFIG)
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables"""
        config_manager_logger.trace("Loading environment variable overrides")
        
        # Check for common override patterns (ensure these are comprehensive)
        # Using a more generic "LEGAL_AI_" prefix is good practice.
        override_prefix = "LEGAL_AI_" 
        
        for key, value in os.environ.items():
            if key.startswith(override_prefix):
                # Store the original env key and also a normalized key for easier access
                normalized_key = key[len(override_prefix):].lower()
                self._environment_overrides[key] = value 
                self._environment_overrides[normalized_key] = value # For easier direct access by normalized name
                config_manager_logger.trace(f"Environment override detected", parameters={'env_key': key, 'normalized_key': normalized_key})
        
        config_manager_logger.info("Environment overrides loaded", parameters={
            'override_count': len(self._environment_overrides) // 2, # Each override stored twice
            'detected_env_keys': [k for k in self._environment_overrides if k.startswith(override_prefix)]
        })
    
    @detailed_log_function(LogCategory.CONFIG)
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by its attribute name in LegalAISettings, with environment override."""
        config_manager_logger.trace(f"Getting configuration value", parameters={'key': key})
        
        # Check environment overrides first using the normalized key
        normalized_key = key.lower()
        if normalized_key in self._environment_overrides:
            # Attempt to cast to the type of the default or the original setting's type if possible
            original_value_type = None
            if hasattr(self._settings, key):
                original_value_type = type(getattr(self._settings, key))
            elif default is not None:
                original_value_type = type(default)

            env_value_str = self._environment_overrides[normalized_key]
            if original_value_type:
                try:
                    if original_value_type == bool:
                        casted_value = env_value_str.lower() in ['true', '1', 'yes']
                    elif original_value_type == int:
                        casted_value = int(env_value_str)
                    elif original_value_type == float:
                        casted_value = float(env_value_str)
                    elif original_value_type == Path:
                        casted_value = Path(env_value_str)
                    elif original_value_type == list: # Basic list parsing from comma-separated string
                        casted_value = [item.strip() for item in env_value_str.split(',')]
                    else: # str or other
                        casted_value = env_value_str
                    config_manager_logger.trace(f"Using environment override for '{key}'", parameters={'value': casted_value})
                    return casted_value
                except ValueError as e:
                    config_manager_logger.warning(f"Could not cast env override for '{key}' to type {original_value_type}. Returning as string.",
                                                 parameters={'env_value': env_value_str, 'error': str(e)})
                    return env_value_str # Return as string if casting fails
            else: # No type hint, return as string
                 config_manager_logger.trace(f"Using environment override for '{key}' (as string)", parameters={'value': env_value_str})
                 return env_value_str


        # Check if value exists in settings object
        if hasattr(self._settings, key):
            value = getattr(self._settings, key)
            config_manager_logger.trace(f"Configuration value retrieved from settings object", parameters={'key': key, 'value_type': type(value).__name__})
            return value
        
        # Return default if key not found
        config_manager_logger.trace(f"Using default value for '{key}'", parameters={'default_value_type': type(default).__name__})
        return default
    
    @detailed_log_function(LogCategory.CONFIG)
    def set_override(self, key: str, value: Any):
        """Set runtime configuration override. This primarily affects in-memory overrides."""
        config_manager_logger.info(f"Setting runtime configuration override", parameters={'key': key})
        
        # Store with a normalized key for consistency with _load_environment_overrides
        normalized_key = key.lower()
        self._environment_overrides[normalized_key] = value 
        # Also store with the conventional env variable like key for completeness, though `get` uses normalized.
        self._environment_overrides[f"LEGAL_AI_{key.upper()}"] = value
        
        # Clearing specific cache entries can be complex if `get` doesn't use caching.
        # If `_config_cache` was used for transformed values, it would be cleared here.
        # For now, `get` reads directly from settings or env overrides.
        
        config_manager_logger.info(f"Runtime configuration override set", parameters={'key': key, 'new_value_type': type(value).__name__})

    # Methods like get_llm_config, get_database_config, etc., are good.
    # They centralize access to related groups of settings.
    # Ensure they use the `self.get()` method to benefit from overrides and logging.

    @detailed_log_function(LogCategory.CONFIG)
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM provider configuration."""
        config_manager_logger.trace("Retrieving LLM configuration")
        
        provider = self.get('llm_provider', 'ollama') # Default to ollama if not set
        config = {
            'provider': provider,
            'model': self.get('llm_model', 'llama3.2' if provider == 'ollama' else ('grok-3-mini' if provider == 'xai' else 'gpt-3.5-turbo')),
            'temperature': self.get('llm_temperature', 0.7),
            'max_tokens': self.get('llm_max_tokens', 4096),
            'fallback_provider': self.get('fallback_provider', 'ollama'),
            'fallback_model': self.get('fallback_model', 'llama3.2')
        }
        
        if provider == 'ollama':
            config.update({
                'host': self.get('ollama_host', 'http://localhost:11434'),
                'timeout': self.get('ollama_timeout', 60)
            })
        elif provider == 'openai':
            config.update({
                'api_key': self.get('openai_api_key'), # Will be None if not set
                'base_url': self.get('openai_base_url')
            })
        elif provider == 'xai':
            config.update({
                'api_key': self.get('xai_api_key'), # Will be None if not set
                'base_url': self.get('xai_base_url', 'https://api.x.ai/v1'),
                'xai_model': self.get('xai_model', 'grok-3-mini') # Specific xAI model
            })
        
        config_manager_logger.info("LLM configuration retrieved", parameters=config)
        return config

    @detailed_log_function(LogCategory.CONFIG)
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        config_manager_logger.trace("Retrieving database configuration")
        
        config = {
            'sqlite_path': str(self.get('sqlite_path', self._settings.data_dir / "databases/legal_ai.db")),
            'memory_db_path': str(self.get('memory_db_path', self._settings.data_dir / "databases/memory.db")),
            'violations_db_path': str(self.get('violations_db_path', self._settings.data_dir / "databases/violations.db")),
            'neo4j_uri': self.get('neo4j_uri', "bolt://localhost:7687"),
            'neo4j_user': self.get('neo4j_user', "neo4j"),
            'neo4j_password': self.get('neo4j_password', "neo4j"), # Default pass, should be in .env
            'neo4j_database': self.get('neo4j_database', "neo4j")
        }
        
        config_manager_logger.info("Database configuration retrieved", parameters=config)
        return config

    @detailed_log_function(LogCategory.CONFIG)
    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration."""
        config_manager_logger.trace("Retrieving vector store configuration")
        
        config = {
            'type': self.get('vector_store_type', 'hybrid'),
            'embedding_model': self.get('embedding_model', "all-MiniLM-L6-v2"),
            'embedding_dim': self.get('embedding_dim', Constants.Performance.EMBEDDING_DIMENSION),
            'faiss_index_path': str(self.get('faiss_index_path', self._settings.data_dir / "vectors/faiss_index.bin")),
            'faiss_metadata_path': str(self.get('faiss_metadata_path', self._settings.data_dir / "vectors/faiss_metadata.json")),
            'lance_db_path': str(self.get('lance_db_path', self._settings.data_dir / "vectors/lancedb")),
            'lance_table_name': self.get('lance_table_name', "documents")
        }
        
        config_manager_logger.info("Vector store configuration retrieved", parameters=config)
        return config

    @detailed_log_function(LogCategory.CONFIG)
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        config_manager_logger.trace("Retrieving security configuration")
        
        config = {
            'enable_data_encryption': self.get('enable_data_encryption', False),
            'encryption_key_path': self.get('encryption_key_path'), # Path or None
            'rate_limit_per_minute': self.get('rate_limit_per_minute', Constants.Security.RATE_LIMIT_PER_MINUTE),
            'enable_request_logging': self.get('enable_request_logging', True),
            'allowed_directories': self.get('allowed_directories', [str(self._settings.documents_dir)]) # Example
        }
        
        config_manager_logger.info("Security configuration retrieved", parameters=config)
        return config

    @detailed_log_function(LogCategory.CONFIG)
    def get_processing_config(self) -> Dict[str, Any]:
        """Get document processing configuration."""
        config_manager_logger.trace("Retrieving processing configuration")
        
        config = {
            'supported_formats': self.get('supported_formats', ['.pdf', '.docx', '.txt', '.md']),
            'max_file_size_mb': self.get('max_file_size_mb', Constants.Document.MAX_DOCUMENT_SIZE_MB),
            'chunk_size': self.get('chunk_size', Constants.Size.DEFAULT_CHUNK_SIZE_CHARS),
            'chunk_overlap': self.get('chunk_overlap', Constants.Size.CHUNK_OVERLAP_CHARS),
            'max_concurrent_documents': self.get('max_concurrent_documents', Constants.Performance.MAX_CONCURRENT_DOCUMENTS),
            'batch_size': self.get('batch_size', Constants.Performance.DEFAULT_BATCH_SIZE),
            'enable_auto_tagging': self.get('enable_auto_tagging', True),
            'auto_tag_confidence_threshold': self.get('auto_tag_confidence_threshold', Constants.Document.AUTO_TAG_CONFIDENCE_THRESHOLD)
        }
        
        config_manager_logger.info("Processing configuration retrieved", parameters=config)
        return config

    @detailed_log_function(LogCategory.CONFIG)
    def get_directories(self) -> Dict[str, Path]:
        """Get system directories, ensuring they exist."""
        config_manager_logger.trace("Retrieving system directories")
        
        # These paths are now set in LegalAISettings.__init__ relative to base_dir
        directories = {
            'base_dir': self._settings.base_dir,
            'data_dir': self._settings.data_dir,
            'documents_dir': self._settings.documents_dir,
            'models_dir': self._settings.models_dir,
            'logs_dir': self._settings.logs_dir,
            # Add other important dirs if they are part of settings
            'faiss_index_dir': self._settings.faiss_index_path.parent,
            'lance_db_dir': self._settings.lance_db_path, # LanceDB path is a directory
            'sqlite_db_dir': self._settings.sqlite_path.parent,
        }
        
        for name, path_obj in directories.items():
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                config_manager_logger.error(f"Failed to create or verify directory '{name}'", 
                                           parameters={'path': str(path_obj)}, exception=e)
                # Depending on severity, you might want to raise an error here for critical dirs
        
        config_manager_logger.info("System directories retrieved and verified/created", 
                                  parameters={name: str(path) for name, path in directories.items()})
        return directories
    
    @detailed_log_function(LogCategory.CONFIG)
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and return status."""
        config_manager_logger.info("Validating system configuration")
        
        validation_results: Dict[str, Any] = {
            'overall_valid': True, # Corrected key
            'warnings': [],
            'errors': [],
            'checks_summary': {} # Corrected key
        }
        
        # Check LLM configuration
        llm_config = self.get_llm_config()
        if llm_config['provider'] in ['openai', 'xai'] and not llm_config.get('api_key'):
            msg = f"API key missing for LLM provider: {llm_config['provider']}"
            validation_results['warnings'].append(msg)
            validation_results['checks_summary']['llm_api_key'] = f"Missing for {llm_config['provider']}"
        else:
            validation_results['checks_summary']['llm_api_key'] = "OK"
        
        # Check directories
        directories = self.get_directories() # This already attempts to create them
        for name, path_obj in directories.items():
            if not path_obj.exists() or not path_obj.is_dir():
                # This case should be rare if get_directories() works, but good to double check critical ones
                if name in ['data_dir', 'logs_dir']: # Critical dirs
                    msg = f"Critical directory missing or not a directory: {name} ({str(path_obj)})"
                    validation_results['errors'].append(msg)
                    validation_results['checks_summary'][f'directory_{name}'] = "Error: Missing"
                    validation_results['overall_valid'] = False
                else:
                    msg = f"Directory missing or not a directory: {name} ({str(path_obj)})"
                    validation_results['warnings'].append(msg)
                    validation_results['checks_summary'][f'directory_{name}'] = "Warning: Missing"
            else:
                validation_results['checks_summary'][f'directory_{name}'] = "OK"
        
        # Check database configuration (basic check for paths)
        db_config = self.get_database_config()
        sqlite_path_obj = Path(db_config['sqlite_path'])
        if not sqlite_path_obj.parent.exists():
            msg = f"SQLite database directory missing: {str(sqlite_path_obj.parent)}"
            validation_results['warnings'].append(msg)
            validation_results['checks_summary']['sqlite_dir'] = "Warning: Missing Parent Dir"
        else:
            validation_results['checks_summary']['sqlite_dir'] = "OK (Parent Dir Exists)"

        if validation_results['errors']: # If any errors, overall is not valid
            validation_results['overall_valid'] = False

        config_manager_logger.info("Configuration validation complete", parameters=validation_results)
        return validation_results
    
    @detailed_log_function(LogCategory.CONFIG)
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all configuration settings as a dictionary."""
        config_manager_logger.trace("Retrieving all configuration settings")
        
        settings_dict = {}
        # Iterate over fields defined in LegalAISettings model if using Pydantic V2
        if hasattr(self._settings, 'model_fields'): # Pydantic V2
            for key in self._settings.model_fields.keys():
                value = self.get(key) # Use self.get to ensure overrides are applied
                if isinstance(value, Path): value = str(value)
                elif isinstance(value, Enum): value = value.value
                settings_dict[key] = value
        elif hasattr(self._settings, '__fields__'): # Pydantic V1
             for key in self._settings.__fields__.keys():
                value = self.get(key)
                if isinstance(value, Path): value = str(value)
                elif isinstance(value, Enum): value = value.value
                settings_dict[key] = value
        else: # Fallback for non-Pydantic or very old Pydantic
            for key in dir(self._settings):
                if not key.startswith('_') and not callable(getattr(self._settings, key)):
                    value = self.get(key)
                    if isinstance(value, Path): value = str(value)
                    elif isinstance(value, Enum): value = value.value
                    settings_dict[key] = value

        config_manager_logger.info("All settings retrieved", parameters={'setting_count': len(settings_dict)})
        return settings_dict

    async def initialize(self) -> 'ConfigurationManager': # For service container compatibility
        """Async initialization hook (currently synchronous)."""
        config_manager_logger.info("ConfigurationManager (async) initialize called.")
        # Current implementation is synchronous, so nothing async to do here.
        # If future versions load config from async sources, this would be used.
        return self

    def health_check(self) -> Dict[str, Any]: # For service container compatibility
        """Performs a health check on the configuration."""
        config_manager_logger.debug("Performing configuration health check.")
        validation = self.validate_configuration()
        return {
            "status": "healthy" if validation['overall_valid'] else "degraded",
            "valid": validation['overall_valid'],
            "warnings": len(validation['warnings']),
            "errors": len(validation['errors']),
            "timestamp": datetime.now().isoformat()
        }

# Global factory function for service container
def create_configuration_manager(custom_settings_instance: Optional[LegalAISettings] = None) -> ConfigurationManager:
    """Factory function to create ConfigurationManager for service container."""
    config_manager_logger.info("Factory: Creating ConfigurationManager instance.")
    manager = ConfigurationManager(custom_settings_instance=custom_settings_instance)
    config_manager_logger.info("Factory: ConfigurationManager instance created successfully.")
    return manager