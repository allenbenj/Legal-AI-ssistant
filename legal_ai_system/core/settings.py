# legal_ai_system/config/settings.py
"""
Enhanced Configuration Management for Legal AI System
Consolidated from multiple sources with Claude Review requirements
Uses centralized constants to eliminate magic numbers.
"""

from pathlib import Path
from typing import List, Optional, Any, Union, cast

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    try:
        from pydantic import BaseSettings, Field  # type: ignore[no-redef]
    except ImportError:
        # Fallback for missing pydantic
        class BaseSettings:  # type: ignore[no-redef]
            """Minimal fallback implementation of Pydantic's ``BaseSettings``."""

            _default_fields: dict[str, Any] = {}

            def __init__(self, **data: Any) -> None:
                for key, value in data.items():
                    setattr(self, key, value)

            def __getattr__(self, name: str) -> Any:
                """Return default field value if available."""
                defaults = getattr(self.__class__, "_default_fields", {})
                if name in defaults:
                    return defaults[name]
                raise AttributeError(
                    f"{type(self).__name__!s} object has no attribute {name!r}"
                )

        def Field(default: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
            return default

        # Add default fields for fallback BaseSettings
        BaseSettings._default_fields = {
            "app_name": "Legal AI Assistant",
            "version": "2.0.0",
            "debug": False,
            "log_level": "INFO",
            "base_dir": Path(__file__)
            .resolve()
            .parent.parent,  # Points to legal_ai_system/
            "data_dir": Path(__file__).resolve().parent.parent / "storage",
            "documents_dir": Path(__file__).resolve().parent.parent
            / "storage/documents",
            "models_dir": Path(__file__).resolve().parent.parent / "models",
            "logs_dir": Path(__file__).resolve().parent.parent / "logs",
            "llm_provider": "xai",
            "llm_model": "grok-3-mini",
            "llm_temperature": 0.7,
            "llm_max_tokens": 4096,
            "ollama_host": "http://localhost:11434",
            "ollama_timeout": 60,
            "openai_api_key": None,
            "openai_base_url": None,
            "xai_api_key": None,
            "xai_base_url": "https://api.x.ai/v1",
            "xai_model": "grok-3-mini",
            "fallback_provider": "ollama",
            "fallback_model": "llama3.2",
            "api_base_url": "http://localhost:8000",
            "vector_store_type": "hybrid",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "faiss_index_path": Path(__file__).resolve().parent.parent
            / "storage/vectors/faiss_index.bin",
            "faiss_metadata_path": Path(__file__).resolve().parent.parent
            / "storage/vectors/faiss_metadata.json",
            "document_index_path": Path(__file__).resolve().parent.parent
            / "storage/vectors/document_index.faiss",
            "entity_index_path": Path(__file__).resolve().parent.parent
            / "storage/vectors/entity_index.faiss",
            "lance_db_path": Path(__file__).resolve().parent.parent
            / "storage/vectors/lancedb",
            "lance_table_name": "documents",
            "sqlite_path": Path(__file__).resolve().parent.parent
            / "storage/databases/legal_ai.db",
            "memory_db_path": Path(__file__).resolve().parent.parent
            / "storage/databases/memory.db",
            "violations_db_path": Path(__file__).resolve().parent.parent
            / "storage/databases/violations.db",
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "CaseDBMS",
            "neo4j_database": "neo4j",
            "supported_formats": [".pdf", ".docx", ".txt", ".md"],
            "max_file_size_mb": 100,
            "tesseract_path": None,
            "ocr_languages": ["eng"],
            "chunk_size": 1000,
            "chunk_overlap": 200,  # Default from Constants.Size
            "enable_auto_tagging": True,
            "auto_tag_confidence_threshold": 0.7,  # Default from Constants.Document
            "tag_history_path": Path(__file__).resolve().parent.parent
            / "storage/tag_history.json",
            "feedback_learning_enabled": True,
            "min_feedback_samples": 50,  # Default from Constants.Performance
            "enable_file_watching": True,
            "watch_directories": [
                str(Path(__file__).resolve().parent.parent / "storage/documents/inbox"),
                str(Path(__file__).resolve().parent.parent / "storage/documents/queue"),
            ],
            "watch_recursive": True,
            "session_timeout_minutes": 30,
            "max_sessions": 10,  # Default from Constants.Time & Constants.Security
            "max_context_tokens": 32000,
            "context_priority_decay": 0.9,  # Default from Constants.Size
            "auto_summarize_enabled": True,
            "auto_summarize_threshold": 5000,
            "summary_max_length": 500,  # Default from Constants.Size
            "default_jurisdiction": "US",
            "jurisdiction_hierarchy": ["Federal", "State", "County", "Municipal"],
            "citation_formats": ["bluebook", "alwd", "mla", "apa"],
            "violation_confidence_threshold": 0.8,
            "enable_cross_case_analysis": True,
            "gui_theme": "light",
            "window_width": 1400,
            "window_height": 900,
            "gui_update_interval_ms": 100,
            "max_gui_log_lines": 1000,
            "max_concurrent_documents": 3,
            "batch_size": 10,  # Default from Constants.Performance
            "enable_embedding_cache": True,
            "cache_ttl_hours": 24,
            "max_cache_size_mb": 500,  # Default from Constants.Time & Constants.Size
            "rate_limit_per_minute": 100,
            "enable_request_logging": True,  # Default from Constants.Security
            "enable_data_encryption": False,
            "encryption_key_path": None,
            "test_data_dir": Path(__file__).resolve().parent.parent / "tests/data",
            "enable_test_mode": False,
            "enable_agent_debugging": False,
            "save_intermediate_results": False,
        }


# Import centralized constants
try:
    from .constants import Constants
except ImportError:
    # Fallback for when constants module is not available directly (e.g. script execution)
    from legal_ai_system.core.constants import Constants


class LegalAISettings(BaseSettings):
    """Comprehensive settings for the Legal AI System"""

    # =================== CORE SYSTEM ===================
    app_name: str = Field(default="Legal AI Assistant", env="APP_NAME")
    version: str = Field(default="2.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    api_base_url: str = Field(default="http://localhost:8000", env="API_BASE_URL")

    # =================== DIRECTORIES ===================
    # Ensure base_dir points to the root of the 'legal_ai_system' project
    base_dir: Path = Field(
        default=Path(__file__).resolve().parent.parent, env="BASE_DIR"
    )
    data_dir: Optional[Path] = Field(default=None, env="DATA_DIR")
    documents_dir: Optional[Path] = Field(default=None, env="DOCUMENTS_DIR")
    models_dir: Optional[Path] = Field(default=None, env="MODELS_DIR")
    logs_dir: Optional[Path] = Field(default=None, env="LOGS_DIR")
    frontend_dist_path: Optional[Path] = Field(
        default=None, env="FRONTEND_DIST_PATH"
    )

    # Auto-create directories
    def __init__(self, **data: Any):
        super().__init__(**data)
        # Set default paths if not provided via environment variables
        if self.data_dir is None:
            self.data_dir = self.base_dir / "storage"
        if self.documents_dir is None:
            self.documents_dir = self.base_dir / "storage" / "documents"
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
        if self.logs_dir is None:
            self.logs_dir = self.base_dir / "logs"
        if self.frontend_dist_path is None:
            self.frontend_dist_path = self.base_dir / "frontend" / "dist"

        # Set path defaults that depend on other fields
        if self.faiss_index_path is None:
            self.faiss_index_path = self.data_dir / "vectors" / "faiss_index.bin"
        if self.faiss_metadata_path is None:
            self.faiss_metadata_path = self.data_dir / "vectors" / "faiss_metadata.json"
        if self.document_index_path is None:
            self.document_index_path = self.data_dir / "vectors" / "document_index.faiss"
        if self.entity_index_path is None:
            self.entity_index_path = self.data_dir / "vectors" / "entity_index.faiss"
        if self.lance_db_path is None:
            self.lance_db_path = self.data_dir / "vectors" / "lancedb"
        if self.sqlite_path is None:
            self.sqlite_path = self.data_dir / "databases" / "legal_ai.db"
        if self.memory_db_path is None:
            self.memory_db_path = self.data_dir / "databases" / "memory.db"
        if self.violations_db_path is None:
            self.violations_db_path = self.data_dir / "databases" / "violations.db"

        # Set embedding dimension from constants if available
        if self.embedding_dim == 384:  # Default fallback value
            try:
                self.embedding_dim = Constants.Performance.EMBEDDING_DIMENSION
            except Exception:
                self.embedding_dim = 384  # Keep fallback

        # Set values from constants if available
        try:
            if self.max_file_size_mb is None:
                self.max_file_size_mb = Constants.Document.MAX_DOCUMENT_SIZE_MB
            if self.chunk_size is None:
                self.chunk_size = Constants.Size.DEFAULT_CHUNK_SIZE_CHARS
            if self.chunk_overlap is None:
                self.chunk_overlap = Constants.Size.CHUNK_OVERLAP_CHARS
            if self.auto_tag_confidence_threshold is None:
                self.auto_tag_confidence_threshold = (
                    Constants.Document.AUTO_TAG_CONFIDENCE_THRESHOLD
                )
            if self.min_feedback_samples is None:
                self.min_feedback_samples = Constants.Performance.MIN_TRAINING_SAMPLES
        except:
            # Fallback values if constants not available
            if self.max_file_size_mb is None:
                self.max_file_size_mb = 50
            if self.chunk_size is None:
                self.chunk_size = 1000
            if self.chunk_overlap is None:
                self.chunk_overlap = 200
            if self.auto_tag_confidence_threshold is None:
                self.auto_tag_confidence_threshold = 0.7
            if self.min_feedback_samples is None:
                self.min_feedback_samples = 10

        for dir_path_attr in ["data_dir", "documents_dir", "models_dir", "logs_dir"]:
            dir_path = getattr(self, dir_path_attr)
            if isinstance(dir_path, Path):  # Check if it's a Path object
                dir_path.mkdir(parents=True, exist_ok=True)
            else:  # It might be a string from fallback
                Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Update specific paths that depend on data_dir
        self.faiss_index_path = self.data_dir / "vectors/faiss_index.bin"
        self.faiss_metadata_path = self.data_dir / "vectors/faiss_metadata.json"
        self.document_index_path = self.data_dir / "vectors/document_index.faiss"
        self.entity_index_path = self.data_dir / "vectors/entity_index.faiss"
        self.lance_db_path = self.data_dir / "vectors/lancedb"
        self.sqlite_path = self.data_dir / "databases/legal_ai.db"
        self.memory_db_path = self.data_dir / "databases/memory.db"
        self.violations_db_path = self.data_dir / "databases/violations.db"
        self.tag_history_path = self.data_dir / "tag_history.json"
        self.test_data_dir = self.base_dir / "tests/data"
        if self.frontend_dist_path is None:
            self.frontend_dist_path = (self.base_dir.parent / "frontend" / "dist").resolve()

    # =================== LLM PROVIDERS ===================
    # Primary LLM
    llm_provider: str = Field(default="xai", env="LLM_PROVIDER")  # ollama, openai, xai
    llm_model: str = Field(default="grok-3-mini", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, env="LLM_MAX_TOKENS")

    # Ollama Settings
    ollama_host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    ollama_timeout: int = Field(default=60, env="OLLAMA_TIMEOUT")

    # OpenAI Settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")

    # xAI/Grok Settings
    xai_api_key: Optional[str] = Field(default=None, env="XAI_API_KEY")
    xai_base_url: str = Field(default="https://api.x.ai/v1", env="XAI_BASE_URL")
    xai_model: str = Field(default="grok-3-mini", env="XAI_MODEL")

    # Fallback Provider
    fallback_provider: str = Field(default="ollama", env="FALLBACK_PROVIDER")
    fallback_model: str = Field(default="llama3.2", env="FALLBACK_MODEL")

    # =================== VECTOR STORAGE ===================
    vector_store_type: str = Field(
        default="hybrid", env="VECTOR_STORE_TYPE"
    )  # faiss, lance, hybrid
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dim: int = Field(
        default=384, env="EMBEDDING_DIM"
    )  # Will be set in __init__

    # FAISS Settings - Path will be updated in __init__
    faiss_index_path: Optional[Path] = Field(default=None, env="FAISS_INDEX_PATH")
    faiss_metadata_path: Optional[Path] = Field(default=None, env="FAISS_METADATA_PATH")
    document_index_path: Optional[Path] = Field(default=None, env="DOCUMENT_INDEX_PATH")
    entity_index_path: Optional[Path] = Field(default=None, env="ENTITY_INDEX_PATH")

    # LanceDB Settings - Path will be updated in __init__
    lance_db_path: Optional[Path] = Field(default=None, env="LANCE_DB_PATH")
    lance_table_name: str = Field(default="documents", env="LANCE_TABLE_NAME")

    # =================== DATABASES ===================
    # SQLite - Paths will be updated in __init__
    sqlite_path: Optional[Path] = Field(default=None, env="SQLITE_PATH")
    memory_db_path: Optional[Path] = Field(default=None, env="MEMORY_DB_PATH")
    violations_db_path: Optional[Path] = Field(default=None, env="VIOLATIONS_DB_PATH")

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(
        default="CaseDBMS", env="NEO4J_PASSWORD"
    )  # Default, should be in .env
    neo4j_database: str = Field(default="neo4j", env="NEO4J_DATABASE")

    # =================== DOCUMENT PROCESSING ===================
    # File Processing
    supported_formats: List[str] = Field(
        default=[".pdf", ".docx", ".txt", ".md"], env="SUPPORTED_FORMATS"
    )
    max_file_size_mb: Optional[int] = Field(default=None, env="MAX_FILE_SIZE_MB")

    # OCR Settings
    tesseract_path: Optional[str] = Field(default=None, env="TESSERACT_PATH")
    ocr_languages: List[str] = Field(default=["eng"], env="OCR_LANGUAGES")

    # Text Processing
    chunk_size: Optional[int] = Field(default=None, env="CHUNK_SIZE")
    chunk_overlap: Optional[int] = Field(default=None, env="CHUNK_OVERLAP")

    # =================== AUTO-TAGGING & INTELLIGENCE ===================
    enable_auto_tagging: bool = Field(default=True, env="ENABLE_AUTO_TAGGING")
    auto_tag_confidence_threshold: Optional[float] = Field(
        default=None, env="AUTO_TAG_THRESHOLD"
    )
    tag_history_path: Optional[Path] = Field(
        default=None, env="TAG_HISTORY_PATH"
    )  # Path updated in __init__

    # Learning System
    feedback_learning_enabled: bool = Field(default=True, env="FEEDBACK_LEARNING")
    min_feedback_samples: Optional[int] = Field(
        default=None, env="MIN_FEEDBACK_SAMPLES"
    )

    # =================== FILE WATCHING ===================
    enable_file_watching: bool = Field(default=True, env="ENABLE_FILE_WATCHING")
    watch_directories: List[str] = Field(
        default=[
            str(
                Path(__file__).resolve().parent.parent / "storage/documents/inbox"
            ),  # Corrected path
            str(
                Path(__file__).resolve().parent.parent / "storage/documents/queue"
            ),  # Corrected path
        ],
        env="WATCH_DIRECTORIES",
    )
    watch_recursive: bool = Field(default=True, env="WATCH_RECURSIVE")

    # =================== MEMORY & CONTEXT ===================
    # Session Management
    session_timeout_minutes: Optional[int] = Field(default=None, env="SESSION_TIMEOUT")
    max_sessions: Optional[int] = Field(default=None, env="MAX_SESSIONS")

    # Context Window
    max_context_tokens: Optional[int] = Field(default=None, env="MAX_CONTEXT_TOKENS")
    context_priority_decay: float = Field(default=0.9, env="CONTEXT_PRIORITY_DECAY")

    # Auto-summarization
    auto_summarize_enabled: bool = Field(default=True, env="AUTO_SUMMARIZE")
    auto_summarize_threshold: Optional[int] = Field(
        default=None, env="AUTO_SUMMARIZE_THRESHOLD"
    )
    summary_max_length: Optional[int] = Field(default=None, env="SUMMARY_MAX_LENGTH")

    # =================== LEGAL SPECIFIC ===================
    # Jurisdiction Settings
    default_jurisdiction: str = Field(default="US", env="DEFAULT_JURISDICTION")
    jurisdiction_hierarchy: List[str] = Field(
        default=["Federal", "State", "County", "Municipal"],
        env="JURISDICTION_HIERARCHY",
    )

    # Citation Processing
    citation_formats: List[str] = Field(
        default=["bluebook", "alwd", "mla", "apa"], env="CITATION_FORMATS"
    )

    # Violation Detection
    violation_confidence_threshold: Optional[float] = Field(
        default=None, env="VIOLATION_THRESHOLD"
    )
    enable_cross_case_analysis: bool = Field(default=True, env="CROSS_CASE_ANALYSIS")

    # =================== GUI SETTINGS ===================
    # Interface
    gui_theme: str = Field(default="light", env="GUI_THEME")  # light, dark, auto
    window_width: int = Field(default=1400, env="WINDOW_WIDTH")
    window_height: int = Field(default=900, env="WINDOW_HEIGHT")

    # Performance
    gui_update_interval_ms: int = Field(default=100, env="GUI_UPDATE_INTERVAL")
    max_gui_log_lines: int = Field(default=1000, env="MAX_GUI_LOG_LINES")

    # =================== PERFORMANCE ===================
    # Processing
    max_concurrent_documents: Optional[int] = Field(
        default=None, env="MAX_CONCURRENT_DOCS"
    )
    batch_size: Optional[int] = Field(default=None, env="BATCH_SIZE")

    # Caching
    enable_embedding_cache: bool = Field(default=True, env="ENABLE_EMBEDDING_CACHE")
    cache_ttl_hours: Optional[int] = Field(default=None, env="CACHE_TTL_HOURS")
    max_cache_size_mb: Optional[int] = Field(default=None, env="MAX_CACHE_SIZE_MB")

    # =================== SECURITY ===================
    # API Security
    rate_limit_per_minute: Optional[int] = Field(
        default=None, env="RATE_LIMIT_PER_MINUTE"
    )
    enable_request_logging: bool = Field(default=True, env="ENABLE_REQUEST_LOGGING")

    # Data Protection
    enable_data_encryption: bool = Field(default=False, env="ENABLE_DATA_ENCRYPTION")
    encryption_key_path: Optional[Path] = Field(default=None, env="ENCRYPTION_KEY_PATH")

    # =================== DEVELOPMENT ===================
    # Testing - Path will be updated in __init__
    test_data_dir: Optional[Path] = Field(default=None, env="TEST_DATA_DIR")
    enable_test_mode: bool = Field(default=False, env="TEST_MODE")

    # Debugging
    enable_agent_debugging: bool = Field(default=False, env="AGENT_DEBUG")
    save_intermediate_results: bool = Field(default=False, env="SAVE_INTERMEDIATE")

    # =================== FRONTEND ===================
    frontend_dist_path: Optional[Path] = Field(
        default=None, env="FRONTEND_DIST_PATH"
    )

    class Config:
        env_file = [".env", ".env.local", ".env.production"]
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Pydantic V2 specific:
        # extra = 'ignore' # If you want to ignore extra fields from .env


# Global settings instance
settings = LegalAISettings()


# Convenience functions
def get_db_url(db_type: str) -> str:
    """Get database connection URL"""
    if db_type == "sqlite":
        return f"sqlite:///{settings.sqlite_path}"
    elif db_type == "neo4j":
        return f"{settings.neo4j_uri}"  # Neo4j uses its own driver, URI is enough
    else:
        raise ValueError(f"Unknown database type: {db_type}")


def get_vector_store_path(store_type: str) -> Path:
    """Get vector store path."""
    if store_type == "faiss":
        return cast(Path, settings.faiss_index_path)  # file path
    elif store_type == "lance":
        return cast(Path, settings.lance_db_path)  # directory path
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")


def is_supported_file(file_path: Union[str, Path]) -> bool:
    """Check if file format is supported."""
    if isinstance(file_path, str):
        file_path = Path(file_path)
    return file_path.suffix.lower() in settings.supported_formats
