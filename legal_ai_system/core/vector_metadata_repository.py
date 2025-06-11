import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover
    GraphDatabase = None  # type: ignore

from .detailed_logging import detailed_log_function, get_detailed_logger, LogCategory
from .unified_exceptions import DatabaseError


class VectorMetadataRepository:
    """PostgreSQL-backed repository for vector metadata."""

    def __init__(
        self,
        dsn: str,
        *,
        min_conn: int = 1,
        max_conn: int = 10,
        neo4j_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dsn = dsn
        self.min_conn = min_conn
        self.max_conn = max_conn
        self.pool: Optional[asyncpg.Pool] = None
        self.logger = get_detailed_logger("VectorMetadataRepo", LogCategory.DATABASE)
        self.neo4j_config = neo4j_config or {}
        self.neo4j_driver = None

    @detailed_log_function(LogCategory.DATABASE)
    async def initialize(self) -> None:
        """Create connection pool and ensure tables exist."""
        try:
            self.pool = await asyncpg.create_pool(
                dsn=self.dsn,
                min_size=self.min_conn,
                max_size=self.max_conn,
            )
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vector_metadata (
                        faiss_id INTEGER,
                        vector_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        content_preview TEXT,
                        vector_norm REAL,
                        dimension INTEGER,
                        created_at_iso TEXT,
                        last_accessed_iso TEXT,
                        access_count INTEGER DEFAULT 0,
                        source_file TEXT,
                        document_type TEXT,
                        tags TEXT,
                        confidence_score REAL DEFAULT 1.0,
                        embedding_model TEXT,
                        custom_metadata JSONB
                    );
                    CREATE TABLE IF NOT EXISTS faiss_id_map (
                        vector_id TEXT PRIMARY KEY,
                        index_target TEXT NOT NULL,
                        faiss_id INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS vector_metadata_audit (
                        audit_id SERIAL PRIMARY KEY,
                        action TEXT NOT NULL,
                        vector_id TEXT,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
        except Exception as e:  # pragma: no cover - initialization
            raise DatabaseError("Failed to initialize metadata repository", cause=e)

        if self.neo4j_config.get("enable") and GraphDatabase:
            try:
                uri = self.neo4j_config.get("uri", "bolt://localhost:7687")
                user = self.neo4j_config.get("user", "neo4j")
                pw = self.neo4j_config.get("password", "neo4j")
                self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, pw))
            except Exception as e:  # pragma: no cover - optional
                self.logger.error("Neo4j connection failed", exception=e)

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
        if self.neo4j_driver:
            self.neo4j_driver.close()

    async def load_metadata_cache(self, limit: int) -> List[Dict[str, Any]]:
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM vector_metadata ORDER BY last_accessed_iso DESC LIMIT $1",
                limit,
            )
            return [dict(r) for r in rows]

    async def load_id_mapping_cache(self) -> List[Dict[str, Any]]:
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT vector_id, index_target, faiss_id FROM faiss_id_map"
            )
            return [dict(r) for r in rows]

    async def find_by_content_hash(self, content_hash: str) -> Optional[str]:
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT vector_id FROM vector_metadata WHERE content_hash = $1 LIMIT 1",
                content_hash,
            )
            return row["vector_id"] if row else None

    async def store_metadata(self, metadata: Dict[str, Any], index_target: str) -> None:
        if not self.pool:
            raise DatabaseError("Repository not initialized")
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO vector_metadata (
                        faiss_id, vector_id, document_id, content_hash,
                        content_preview, vector_norm, dimension,
                        created_at_iso, last_accessed_iso, access_count,
                        source_file, document_type, tags,
                        confidence_score, embedding_model, custom_metadata
                    ) VALUES (
                        $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16
                    ) ON CONFLICT(vector_id) DO UPDATE SET
                        faiss_id=EXCLUDED.faiss_id,
                        document_id=EXCLUDED.document_id,
                        content_hash=EXCLUDED.content_hash,
                        content_preview=EXCLUDED.content_preview,
                        vector_norm=EXCLUDED.vector_norm,
                        dimension=EXCLUDED.dimension,
                        embedding_model=EXCLUDED.embedding_model,
                        custom_metadata=EXCLUDED.custom_metadata,
                        last_accessed_iso=EXCLUDED.last_accessed_iso
                    """,
                    metadata.get("faiss_id"),
                    metadata.get("vector_id"),
                    metadata.get("document_id"),
                    metadata.get("content_hash"),
                    metadata.get("content_preview"),
                    metadata.get("vector_norm"),
                    metadata.get("dimension"),
                    metadata.get("created_at_iso"),
                    metadata.get("last_accessed_iso"),
                    metadata.get("access_count", 0),
                    metadata.get("source_file"),
                    metadata.get("document_type"),
                    metadata.get("tags"),
                    metadata.get("confidence_score", 1.0),
                    metadata.get("embedding_model"),
                    json.dumps(metadata.get("custom_metadata", {})),
                )
                await conn.execute(
                    """
                    INSERT INTO faiss_id_map (vector_id, index_target, faiss_id)
                    VALUES ($1,$2,$3)
                    ON CONFLICT(vector_id) DO UPDATE SET
                        index_target=EXCLUDED.index_target,
                        faiss_id=EXCLUDED.faiss_id
                    """,
                    metadata.get("vector_id"),
                    index_target,
                    metadata.get("faiss_id"),
                )
                await conn.execute(
                    "INSERT INTO vector_metadata_audit(action, vector_id, metadata) VALUES($1,$2,$3)",
                    "upsert",
                    metadata.get("vector_id"),
                    json.dumps(metadata),
                )
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    session.run(
                        "MERGE (v:Vector {id:$id}) SET v.document_id=$doc_id, v.content_hash=$hash",
                        id=metadata.get("vector_id"),
                        doc_id=metadata.get("document_id"),
                        hash=metadata.get("content_hash"),
                    )
            except Exception as e:  # pragma: no cover - neo4j optional
                self.logger.error("Failed to sync vector to Neo4j", exception=e)

    async def get_faiss_id_for_vector_id(self, vector_id: str, index_target: str) -> Optional[int]:
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT faiss_id FROM faiss_id_map WHERE vector_id=$1 AND index_target=$2",
                vector_id,
                index_target,
            )
            return int(row["faiss_id"]) if row else None

    async def get_vector_id_by_faiss_id(self, faiss_id: int, index_target: str) -> Optional[str]:
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT vector_id FROM faiss_id_map WHERE faiss_id=$1 AND index_target=$2",
                faiss_id,
                index_target,
            )
            return row["vector_id"] if row else None

    async def update_access_stats(self, vector_id: str) -> None:
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE vector_metadata
                SET last_accessed_iso=$1, access_count=access_count + 1
                WHERE vector_id=$2
                """,
                datetime.now(timezone.utc).isoformat(),
                vector_id,
            )

    async def update_metadata_fields(self, vector_id: str, updates: Dict[str, Any]) -> None:
        if not self.pool or not updates:
            return
        set_clauses = []
        values = []
        for idx, (key, value) in enumerate(updates.items(), start=1):
            set_clauses.append(f"{key} = ${idx}")
            values.append(value)
        values.append(vector_id)
        query = f"UPDATE vector_metadata SET {', '.join(set_clauses)} WHERE vector_id=${len(values)}"
        async with self.pool.acquire() as conn:
            await conn.execute(query, *values)

    async def delete_vector(self, vector_id: str) -> None:
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM vector_metadata WHERE vector_id=$1",
                    vector_id,
                )
                await conn.execute(
                    "DELETE FROM faiss_id_map WHERE vector_id=$1",
                    vector_id,
                )
                await conn.execute(
                    "INSERT INTO vector_metadata_audit(action, vector_id, metadata) VALUES('delete',$1,NULL)",
                    vector_id,
                )

