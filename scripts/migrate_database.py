#!/usr/bin/env python3
"""Database migration script using EnhancedPersistenceManager."""
import asyncio
from legal_ai_system.core.enhanced_persistence import (
    create_enhanced_persistence_manager,
    ConnectionPool,
)
import os

async def main():
    cfg = {
        "database_url": os.getenv("DATABASE_URL"),
        "redis_url": os.getenv("REDIS_URL"),
    }
    pool = ConnectionPool(cfg.get("database_url"), cfg.get("redis_url"))
    manager = create_enhanced_persistence_manager(
        None,
        config=cfg,
        connection_pool=pool,
    )
    await manager.initialize()
    await manager.close()

if __name__ == '__main__':
    asyncio.run(main())
