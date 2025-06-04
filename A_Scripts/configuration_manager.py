"""
Configuration Manager - Centralized Configuration Management
===========================================================
Provides a service-oriented interface to the Legal AI System configuration.
"""

from typing import Any, Dict, Optional, List
from pathlib import Path
import os

# Import detailed logging system
from .detailed_logging import get_detailed_logger, LogCategory, detailed_log_function

# Import configuration settings
try:
    from ..config.settings import settings, LegalAISettings, get_db_url, get_vector_store_path, is_supported_file
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.settings import settings, LegalAISettings, get_db_url, get_vector_store_path, is_supported_file

# Initialize logger
config_logger = get_detailed_logger("Configuration_Manager", LogCategory.SYSTEM)

class ConfigurationManager:
    """
    Service-oriented configuration manager that provides centralized
    access to all Legal AI System configuration settings.
    """
    
    @detailed_log_function(LogCategory.SYSTEM)
    def __init__(self, custom_settings: Optional[LegalAISettings] = None):
        """Initialize configuration manager with optional custom settings"""
        config_logger.info("=== INITIALIZING CONFIGURATION MANAGER ===")
        
        self._settings = custom_settings or settings
        self._config_cache: Dict[str, Any] = {}
        self._environment_overrides: Dict[str, Any] = {}
        
        # Load environment overrides
        self._load_environment_overrides()
        
        config_logger.info("Configuration manager initialized", parameters={
            'app_name': self._settings.app_name,
            'version': self._settings.version,
            'debug_mode': self._settings.debug,
            'llm_provider': self._settings.llm_provider,
            'vector_store_type': self._settings.vector_store_type,
            'environment_overrides': len(self._environment_overrides)
        })
    
    @detailed_log_function(LogCategory.SYSTEM)
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables"""
        config_logger.trace("Loading environment variable overrides")
        
        # Check for common override patterns
        override_patterns = [
            'LEGAL_AI_',
            'APP_',
            'LLM_',
            'NEO4J_',
            'FAISS_',
            'LANCE_'
        ]
        
        for key, value in os.environ.items():
            for pattern in override_patterns:
                if key.startswith(pattern):
                    self._environment_overrides[key] = value
                    config_logger.trace(f"Environment override detected: {key}")
        
        config_logger.info("Environment overrides loaded", parameters={
            'override_count': len(self._environment_overrides),
            'override_keys': list(self._environment_overrides.keys())
        })
    
    @detailed_log_function(LogCategory.SYSTEM)
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback support"""
        config_logger.trace(f"Getting configuration value: {key}")
        
        # Check environment overrides first
        env_key = f"LEGAL_AI_{key.upper()}"
        if env_key in self._environment_overrides:
            value = self._environment_overrides[env_key]
            config_logger.trace(f"Using environment override for {key}", parameters={'value': str(value)})
            return value
        
        # Check if value exists in settings
        if hasattr(self._settings, key):
            value = getattr(self._settings, key)
            config_logger.trace(f"Configuration value retrieved: {key}", parameters={'value': str(value)})
            return value
        
        # Return default
        config_logger.trace(f"Using default value for {key}", parameters={'default': str(default)})
        return default
    
    @detailed_log_function(LogCategory.SYSTEM)
    def set_override(self, key: str, value: Any):
        """Set runtime configuration override"""
        config_logger.info(f"Setting configuration override: {key}")
        
        self._environment_overrides[f"LEGAL_AI_{key.upper()}"] = value
        
        # Clear cache for this key
        cache_key = f"cached_{key}"
        if cache_key in self._config_cache:
            del self._config_cache[cache_key]
        
        config_logger.info(f"Configuration override set: {key}", parameters={'value': str(value)})
    
    @detailed_log_function(LogCategory.SYSTEM)
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM provider configuration"""
        config_logger.trace("Retrieving LLM configuration")
        
        config = {
            'provider': self.get('llm_provider'),
            'model': self.get('llm_model'),
            'temperature': self.get('llm_temperature'),
            'max_tokens': self.get('llm_max_tokens'),
            'fallback_provider': self.get('fallback_provider'),
            'fallback_model': self.get('fallback_model')
        }
        
        # Add provider-specific configs
        if config['provider'] == 'ollama':
            config.update({
                'host': self.get('ollama_host'),
                'timeout': self.get('ollama_timeout')
            })
        elif config['provider'] == 'openai':
            config.update({
                'api_key': self.get('openai_api_key'),
                'base_url': self.get('openai_base_url')
            })
        elif config['provider'] == 'xai':
            config.update({
                'api_key': self.get('xai_api_key'),
                'base_url': self.get('xai_base_url'),
                'xai_model': self.get('xai_model')
            })
        
        config_logger.info("LLM configuration retrieved", parameters={
            'provider': config['provider'],
            'model': config['model'],
            'has_api_key': bool(config.get('api_key'))
        })
        
        return config
    
    @detailed_log_function(LogCategory.SYSTEM)
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        config_logger.trace("Retrieving database configuration")
        
        config = {
            'sqlite_path': str(self.get('sqlite_path')),
            'memory_db_path': str(self.get('memory_db_path')),
            'violations_db_path': str(self.get('violations_db_path')),
            'neo4j_uri': self.get('neo4j_uri'),
            'neo4j_user': self.get('neo4j_user'),
            'neo4j_password': self.get('neo4j_password'),
            'neo4j_database': self.get('neo4j_database')
        }
        
        config_logger.info("Database configuration retrieved", parameters={
            'sqlite_enabled': bool(config['sqlite_path']),
            'neo4j_enabled': bool(config['neo4j_uri']),
            'neo4j_uri': config['neo4j_uri']
        })
        
        return config
    
    @detailed_log_function(LogCategory.SYSTEM)
    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration"""
        config_logger.trace("Retrieving vector store configuration")
        
        config = {
            'type': self.get('vector_store_type'),
            'embedding_model': self.get('embedding_model'),
            'embedding_dim': self.get('embedding_dim'),
            'faiss_index_path': str(self.get('faiss_index_path')),
            'faiss_metadata_path': str(self.get('faiss_metadata_path')),
            'lance_db_path': str(self.get('lance_db_path')),
            'lance_table_name': self.get('lance_table_name')
        }
        
        config_logger.info("Vector store configuration retrieved", parameters={
            'type': config['type'],
            'embedding_model': config['embedding_model'],
            'embedding_dim': config['embedding_dim']
        })
        
        return config
    
    @detailed_log_function(LogCategory.SYSTEM)
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        config_logger.trace("Retrieving security configuration")
        
        config = {
            'enable_data_encryption': self.get('enable_data_encryption'),
            'encryption_key_path': self.get('encryption_key_path'),
            'rate_limit_per_minute': self.get('rate_limit_per_minute'),
            'enable_request_logging': self.get('enable_request_logging'),
            'allowed_directories': [
                str(self.get('documents_dir')),
                str(self.get('data_dir')),
                str(self.get('models_dir'))
            ]
        }
        
        config_logger.info("Security configuration retrieved", parameters={
            'encryption_enabled': config['enable_data_encryption'],
            'request_logging_enabled': config['enable_request_logging'],
            'allowed_directories_count': len(config['allowed_directories'])
        })
        
        return config
    
    @detailed_log_function(LogCategory.SYSTEM)
    def get_processing_config(self) -> Dict[str, Any]:
        """Get document processing configuration"""
        config_logger.trace("Retrieving processing configuration")
        
        config = {
            'supported_formats': self.get('supported_formats'),
            'max_file_size_mb': self.get('max_file_size_mb'),
            'chunk_size': self.get('chunk_size'),
            'chunk_overlap': self.get('chunk_overlap'),
            'max_concurrent_documents': self.get('max_concurrent_documents'),
            'batch_size': self.get('batch_size'),
            'enable_auto_tagging': self.get('enable_auto_tagging'),
            'auto_tag_confidence_threshold': self.get('auto_tag_confidence_threshold')
        }
        
        config_logger.info("Processing configuration retrieved", parameters={
            'supported_formats': config['supported_formats'],
            'max_file_size_mb': config['max_file_size_mb'],
            'auto_tagging_enabled': config['enable_auto_tagging']
        })
        
        return config
    
    @detailed_log_function(LogCategory.SYSTEM)
    def get_directories(self) -> Dict[str, Path]:
        """Get system directories"""
        config_logger.trace("Retrieving system directories")
        
        directories = {
            'base_dir': Path(self.get('base_dir')),
            'data_dir': Path(self.get('data_dir')),
            'documents_dir': Path(self.get('documents_dir')),
            'models_dir': Path(self.get('models_dir')),
            'logs_dir': Path(self.get('logs_dir'))
        }
        
        # Ensure directories exist
        for name, path in directories.items():
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                config_logger.trace(f"Created directory: {name}", parameters={'path': str(path)})
        
        config_logger.info("System directories retrieved", parameters={
            'directory_count': len(directories),
            'base_dir': str(directories['base_dir'])
        })
        
        return directories
    
    @detailed_log_function(LogCategory.SYSTEM)
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and return status"""
        config_logger.info("Validating system configuration")
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'checks': {}
        }
        
        # Check LLM configuration
        llm_config = self.get_llm_config()
        if llm_config['provider'] in ['openai', 'xai'] and not llm_config.get('api_key'):
            validation_results['warnings'].append(f"No API key configured for {llm_config['provider']}")
            validation_results['checks']['llm_api_key'] = False
        else:
            validation_results['checks']['llm_api_key'] = True
        
        # Check directories
        directories = self.get_directories()
        for name, path in directories.items():
            if not path.exists():
                validation_results['errors'].append(f"Required directory missing: {name} ({path})")
                validation_results['checks'][f'directory_{name}'] = False
                validation_results['valid'] = False
            else:
                validation_results['checks'][f'directory_{name}'] = True
        
        # Check database configuration
        db_config = self.get_database_config()
        sqlite_path = Path(db_config['sqlite_path'])
        if not sqlite_path.parent.exists():
            validation_results['warnings'].append(f"SQLite database directory missing: {sqlite_path.parent}")
            validation_results['checks']['sqlite_dir'] = False
        else:
            validation_results['checks']['sqlite_dir'] = True
        
        config_logger.info("Configuration validation complete", parameters={
            'valid': validation_results['valid'],
            'warning_count': len(validation_results['warnings']),
            'error_count': len(validation_results['errors'])
        })
        
        return validation_results
    
    @detailed_log_function(LogCategory.SYSTEM)
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all configuration settings as dictionary"""
        config_logger.trace("Retrieving all configuration settings")
        
        # Convert settings to dictionary
        settings_dict = {}
        for key in dir(self._settings):
            if not key.startswith('_') and not callable(getattr(self._settings, key)):
                value = getattr(self._settings, key)
                # Convert Path objects to strings for serialization
                if isinstance(value, Path):
                    value = str(value)
                settings_dict[key] = value
        
        config_logger.info("All settings retrieved", parameters={
            'setting_count': len(settings_dict)
        })
        
        return settings_dict

def create_configuration_manager(config: Optional[Dict[str, Any]] = None) -> ConfigurationManager:
    """Factory function to create ConfigurationManager"""
    config_logger.info("Creating ConfigurationManager instance")
    
    custom_settings = None
    if config:
        # Create custom settings instance with overrides
        custom_settings = LegalAISettings()
        for key, value in config.items():
            if hasattr(custom_settings, key):
                setattr(custom_settings, key, value)
    
    manager = ConfigurationManager(custom_settings)
    
    config_logger.info("ConfigurationManager created successfully")
    return manager