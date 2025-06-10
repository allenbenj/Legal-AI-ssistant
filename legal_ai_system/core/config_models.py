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

class SecurityConfig(BaseModel):
    """Security related configuration."""

    enable_data_encryption: bool = Field(..., description="Enable encryption of data")
    encryption_key_path: Optional[Path] = Field(None, description="Path to encryption key")
    rate_limit_per_minute: int = Field(..., description="API rate limit")
    enable_request_logging: bool = Field(..., description="Enable request logging")
    allowed_directories: List[str] = Field(default_factory=list, description="Directories accessible by the system")
