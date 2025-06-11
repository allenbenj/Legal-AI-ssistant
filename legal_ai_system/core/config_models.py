from __future__ import annotations

"""Typed configuration models used across the core services."""

from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """Database related settings."""

    sqlite_path: Path = Field(..., description="Path to SQLite database")
    memory_db_path: Path = Field(..., description="Path to unified memory DB")
    violations_db_path: Path = Field(..., description="Path to violations DB")
    neo4j_uri: str = Field(..., description="Neo4j connection URI")
    neo4j_user: str = Field(..., description="Neo4j username")
    neo4j_password: str = Field(..., description="Neo4j password")
    neo4j_database: str = Field(..., description="Neo4j database name")


class VectorStoreConfig(BaseModel):
    """Configuration for the vector store implementation."""

    type: str = Field(..., description="Vector store type")
    embedding_model: str = Field(..., description="Embedding model name")
    embedding_dim: int = Field(..., description="Embedding vector dimension")
    faiss_index_path: Path = Field(..., description="Path to FAISS index")
    faiss_metadata_path: Path = Field(..., description="Path to FAISS metadata")
    document_index_path: Path = Field(..., description="Path to document index")
    entity_index_path: Path = Field(..., description="Path to entity index")
    lance_db_path: Path = Field(..., description="Directory for LanceDB store")
    lance_table_name: str = Field(..., description="Default LanceDB table name")
    optimization_interval_sec: int = Field(
        3600, description="Interval between automatic optimization runs"
    )
    backup_interval_sec: int = Field(
        86400, description="Interval between automatic backups"
    )
    instances: int = Field(1, description="Number of vector store instances")
    hosts: List[str] = Field(default_factory=list, description="Vector store host URLs")
    load_balancing_strategy: str = Field(
        "round_robin", description="Client-side load balancing strategy"
    )


class SecurityConfig(BaseModel):
    """Security related configuration."""

    enable_data_encryption: bool = Field(..., description="Enable encryption of data")
    encryption_key_path: Optional[Path] = Field(
        None, description="Path to encryption key"
    )
    rate_limit_per_minute: int = Field(..., description="API rate limit")
    enable_request_logging: bool = Field(..., description="Enable request logging")
    allowed_directories: List[str] = Field(
        default_factory=list, description="Directories accessible by the system"
    )


class MLOptimizerConfig(BaseModel):
    """Configuration for :class:`MLOptimizer`."""

    min_samples: int = Field(
        50, description="Minimum samples required for optimization"
    )
    similarity_threshold: float = Field(
        0.8, description="Threshold for document similarity"
    )
    max_optimization_age_hours: int = Field(
        24, description="Maximum age of data used for optimization"
    )


class UnifiedMemoryManagerConfig(BaseModel):
    """Settings for :class:`UnifiedMemoryManager`."""

    db_path: Path = Field(
        "./storage/databases/unified_memory.db",
        description="Path to the memory database",
    )
    max_context_tokens: int = Field(
        32_000, description="Maximum context tokens stored per session"
    )


class OptimizedVectorStoreConfig(BaseModel):
    """Configuration options for :class:`OptimizedVectorStore`."""

    storage_path: Path = Field(
        "./storage/vectors", description="Vector store directory"
    )
    embedding_model_name: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", description="Default embedding model"
    )
    default_index_type: str = Field("HNSW", description="Index type for FAISS")
    enable_gpu_faiss: bool = Field(False, description="Enable GPU acceleration")
    index_params: Optional[dict] = Field(
        None,
        description="Additional parameters passed to the underlying index implementation",
    )
