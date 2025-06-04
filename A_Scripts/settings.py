"""
Enhanced Configuration Management for Legal AI System
Consolidated from multiple sources with Claude Review requirements
Uses centralized constants to eliminate magic numbers.
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    try:
        from pydantic import BaseSettings, Field
    except ImportError:
        # Fallback for missing pydantic
        class BaseSettings:
            pass
        def Field(default, **kwargs):
            return default
import os

# Import centralized constants
try:
    from ..core.constants import Constants
except ImportError:
    # Fallback for when constants module is not available
    class Constants:
        class Time:
            SESSION_TIMEOUT_MINUTES = 30
        class Size:
            DEFAULT_CACHE_SIZE_ITEMS = 1000
            MAX_CONTEXT_TOKENS = 32_000
            SUMMARY_MAX_LENGTH_CHARS = 500
            AUTO_SUMMARIZE_THRESHOLD_CHARS = 5000
            DEFAULT_CHUNK_SIZE_CHARS = 1000
            CHUNK_OVERLAP_CHARS = 200
        class Security:
            RATE_LIMIT_PER_MINUTE = 100
        class Performance:
            MAX_CONCURRENT_DOCUMENTS = 3
            DEFAULT_BATCH_SIZE = 10
        class Document:
            AUTO_TAG_CONFIDENCE_THRESHOLD = 0.7

class LegalAISettings(BaseSettings):
    """Comprehensive settings for the Legal AI System"""
    
    # =================== CORE SYSTEM ===================
    app_name: str = Field("Legal AI Assistant", env="APP_NAME")
    version: str = Field("2.0.0", env="APP_VERSION")
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # =================== DIRECTORIES ===================
    base_dir: Path = Field(Path(__file__).parent.parent, env="BASE_DIR")
    data_dir: Path = Field(base_dir / "storage", env="DATA_DIR")
    documents_dir: Path = Field(base_dir / "storage/documents", env="DOCUMENTS_DIR")
    models_dir: Path = Field(base_dir / "models", env="MODELS_DIR")
    logs_dir: Path = Field(base_dir / "logs", env="LOGS_DIR")
    
    # Auto-create directories
    def __post_init__(self):
        for dir_path in [self.data_dir, self.documents_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # =================== LLM PROVIDERS ===================
    # Primary LLM
    llm_provider: str = Field("xai", env="LLM_PROVIDER")  # ollama, openai, xai
    llm_model: str = Field("grok-3-mini", env="LLM_MODEL")
    llm_temperature: float = Field(0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(4096, env="LLM_MAX_TOKENS")
    
    # Ollama Settings
    ollama_host: str = Field("http://localhost:11434", env="OLLAMA_HOST")
    ollama_timeout: int = Field(60, env="OLLAMA_TIMEOUT")
    
    # OpenAI Settings
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(None, env="OPENAI_BASE_URL")
    
    # xAI/Grok Settings
    xai_api_key: Optional[str] = Field(None, env="XAI_API_KEY")
    xai_base_url: str = Field("https://api.x.ai/v1", env="XAI_BASE_URL")
    xai_model: str = Field("grok-3-mini", env="XAI_MODEL")
    
    # Fallback Provider
    fallback_provider: str = Field("ollama", env="FALLBACK_PROVIDER")
    fallback_model: str = Field("llama3.2", env="FALLBACK_MODEL")
    
    # =================== VECTOR STORAGE ===================
    vector_store_type: str = Field("hybrid", env="VECTOR_STORE_TYPE")  # faiss, lance, hybrid
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dim: int = Field(384, env="EMBEDDING_DIM")
    
    # FAISS Settings
    faiss_index_path: Path = Field(data_dir / "vectors/faiss_index.bin", env="FAISS_INDEX_PATH")
    faiss_metadata_path: Path = Field(data_dir / "vectors/faiss_metadata.json", env="FAISS_METADATA_PATH")
    
    # LanceDB Settings
    lance_db_path: Path = Field(data_dir / "vectors/lancedb", env="LANCE_DB_PATH")
    lance_table_name: str = Field("documents", env="LANCE_TABLE_NAME")
    
    # =================== DATABASES ===================
    # SQLite
    sqlite_path: Path = Field(data_dir / "databases/legal_ai.db", env="SQLITE_PATH")
    memory_db_path: Path = Field(data_dir / "databases/memory.db", env="MEMORY_DB_PATH")
    violations_db_path: Path = Field(data_dir / "databases/violations.db", env="VIOLATIONS_DB_PATH")
    
    # Neo4j
    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field("neo4j", env="NEO4J_USER")
    neo4j_password: str = Field("CaseDBMS", env="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", env="NEO4J_DATABASE")
    
    # =================== DOCUMENT PROCESSING ===================
    # File Processing
    supported_formats: List[str] = Field(['.pdf', '.docx', '.txt', '.md'], env="SUPPORTED_FORMATS")
    max_file_size_mb: int = Field(100, env="MAX_FILE_SIZE_MB")
    
    # OCR Settings
    tesseract_path: Optional[str] = Field(None, env="TESSERACT_PATH")
    ocr_languages: List[str] = Field(['eng'], env="OCR_LANGUAGES")
    
    # Text Processing
    chunk_size: int = Field(Constants.Size.DEFAULT_CHUNK_SIZE_CHARS, env="CHUNK_SIZE")
    chunk_overlap: int = Field(Constants.Size.CHUNK_OVERLAP_CHARS, env="CHUNK_OVERLAP")
    
    # =================== AUTO-TAGGING & INTELLIGENCE ===================
    enable_auto_tagging: bool = Field(True, env="ENABLE_AUTO_TAGGING")
    auto_tag_confidence_threshold: float = Field(Constants.Document.AUTO_TAG_CONFIDENCE_THRESHOLD, env="AUTO_TAG_THRESHOLD")
    tag_history_path: Path = Field(data_dir / "tag_history.json", env="TAG_HISTORY_PATH")
    
    # Learning System
    feedback_learning_enabled: bool = Field(True, env="FEEDBACK_LEARNING")
    min_feedback_samples: int = Field(Constants.Performance.MIN_TRAINING_SAMPLES, env="MIN_FEEDBACK_SAMPLES")
    
    # =================== FILE WATCHING ===================
    enable_file_watching: bool = Field(True, env="ENABLE_FILE_WATCHING")
    watch_directories: List[str] = Field([
        str(documents_dir / "inbox"),
        str(documents_dir / "queue")
    ], env="WATCH_DIRECTORIES")
    watch_recursive: bool = Field(True, env="WATCH_RECURSIVE")
    
    # =================== MEMORY & CONTEXT ===================
    # Session Management
    session_timeout_minutes: int = Field(Constants.Time.SESSION_TIMEOUT_MINUTES, env="SESSION_TIMEOUT")
    max_sessions: int = Field(Constants.Security.MAX_CONCURRENT_SESSIONS, env="MAX_SESSIONS")
    
    # Context Window
    max_context_tokens: int = Field(Constants.Size.MAX_CONTEXT_TOKENS, env="MAX_CONTEXT_TOKENS")
    context_priority_decay: float = Field(0.9, env="CONTEXT_PRIORITY_DECAY")
    
    # Auto-summarization
    auto_summarize_enabled: bool = Field(True, env="AUTO_SUMMARIZE")
    auto_summarize_threshold: int = Field(Constants.Size.AUTO_SUMMARIZE_THRESHOLD_CHARS, env="AUTO_SUMMARIZE_THRESHOLD")
    summary_max_length: int = Field(Constants.Size.SUMMARY_MAX_LENGTH_CHARS, env="SUMMARY_MAX_LENGTH")
    
    # =================== LEGAL SPECIFIC ===================
    # Jurisdiction Settings
    default_jurisdiction: str = Field("US", env="DEFAULT_JURISDICTION")
    jurisdiction_hierarchy: List[str] = Field([
        "Federal", "State", "County", "Municipal"
    ], env="JURISDICTION_HIERARCHY")
    
    # Citation Processing
    citation_formats: List[str] = Field([
        "bluebook", "alwd", "mla", "apa"
    ], env="CITATION_FORMATS")
    
    # Violation Detection
    violation_confidence_threshold: float = Field(0.8, env="VIOLATION_THRESHOLD")
    enable_cross_case_analysis: bool = Field(True, env="CROSS_CASE_ANALYSIS")
    
    # =================== GUI SETTINGS ===================
    # Interface
    gui_theme: str = Field("light", env="GUI_THEME")  # light, dark, auto
    window_width: int = Field(1400, env="WINDOW_WIDTH")
    window_height: int = Field(900, env="WINDOW_HEIGHT")
    
    # Performance
    gui_update_interval_ms: int = Field(100, env="GUI_UPDATE_INTERVAL")
    max_gui_log_lines: int = Field(1000, env="MAX_GUI_LOG_LINES")
    
    # =================== PERFORMANCE ===================
    # Processing
    max_concurrent_documents: int = Field(Constants.Performance.MAX_CONCURRENT_DOCUMENTS, env="MAX_CONCURRENT_DOCS")
    batch_size: int = Field(Constants.Performance.DEFAULT_BATCH_SIZE, env="BATCH_SIZE")
    
    # Caching
    enable_embedding_cache: bool = Field(True, env="ENABLE_EMBEDDING_CACHE")
    cache_ttl_hours: int = Field(Constants.Time.DEFAULT_CACHE_TTL_HOURS, env="CACHE_TTL_HOURS")
    max_cache_size_mb: int = Field(Constants.Size.MAX_CACHE_SIZE_MB, env="MAX_CACHE_SIZE_MB")
    
    # =================== SECURITY ===================
    # API Security
    rate_limit_per_minute: int = Field(Constants.Security.RATE_LIMIT_PER_MINUTE, env="RATE_LIMIT_PER_MINUTE")
    enable_request_logging: bool = Field(True, env="ENABLE_REQUEST_LOGGING")
    
    # Data Protection
    enable_data_encryption: bool = Field(False, env="ENABLE_DATA_ENCRYPTION")
    encryption_key_path: Optional[Path] = Field(None, env="ENCRYPTION_KEY_PATH")
    
    # =================== DEVELOPMENT ===================
    # Testing
    test_data_dir: Path = Field(base_dir / "tests/data", env="TEST_DATA_DIR")
    enable_test_mode: bool = Field(False, env="TEST_MODE")
    
    # Debugging
    enable_agent_debugging: bool = Field(False, env="AGENT_DEBUG")
    save_intermediate_results: bool = Field(False, env="SAVE_INTERMEDIATE")
    
    class Config:
        env_file = ['.env', '.env.local', '.env.production']
        env_file_encoding = 'utf-8'
        case_sensitive = False

# Global settings instance
settings = LegalAISettings()

# Convenience functions
def get_db_url(db_type: str) -> str:
    """Get database connection URL"""
    if db_type == "sqlite":
        return f"sqlite:///{settings.sqlite_path}"
    elif db_type == "neo4j":
        return f"{settings.neo4j_uri}"
    else:
        raise ValueError(f"Unknown database type: {db_type}")

def get_vector_store_path(store_type: str) -> Path:
    """Get vector store path"""
    if store_type == "faiss":
        return settings.faiss_index_path
    elif store_type == "lance":
        return settings.lance_db_path
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")

def is_supported_file(file_path: Path) -> bool:
    """Check if file format is supported"""
    return file_path.suffix.lower() in settings.supported_formats