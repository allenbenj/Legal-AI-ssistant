#!/usr/bin/env python3
"""Database migration script using :class:`EnhancedPersistenceManager`."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict
from urllib.parse import urlparse

from legal_ai_system.core.enhanced_persistence import (
    create_enhanced_persistence_manager,
    ConnectionPool,
    EnhancedPersistenceManager,
)
from legal_ai_system.scripts.init_postgres_db import ensure_database_exists


async def run_migrations(manager: EnhancedPersistenceManager) -> None:
    """Execute database migrations.

    Schema creation occurs during ``manager.initialize()``. This function
    acts as an extension point for future migrations that require data
    transformations.
    """

    # Placeholder for future migration logic.
    # Add migration steps here when schema changes require it.
    pass


async def main() -> None:
    """Initialize the persistence layer, run migrations, then close."""

    config: Dict[str, Any] = {
        "database_url": os.getenv("DATABASE_URL"),
        "redis_url": os.getenv("REDIS_URL"),
    }

    if config["database_url"]:
        parsed = urlparse(config["database_url"])
        await ensure_database_exists(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            user=parsed.username or os.getenv("POSTGRES_USER", "postgres"),
            password=parsed.password or os.getenv("POSTGRES_PASSWORD", ""),
            database=parsed.path.lstrip("/") or os.getenv("POSTGRES_DB", "legal_ai"),
            admin_db=os.getenv("POSTGRES_ADMIN_DB", "postgres"),
        )

    pool = ConnectionPool(
        config.get("database_url"),
        config.get("redis_url"),
    )
    manager = create_enhanced_persistence_manager(
        config=config,
        connection_pool=pool,
    )

    await manager.initialize()
    await run_migrations(manager)
    await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
