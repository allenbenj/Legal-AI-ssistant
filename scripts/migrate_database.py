#!/usr/bin/env python3
"""Database migration script using EnhancedPersistenceManager."""
import asyncio
from legal_ai_system.core.enhanced_persistence import (
    create_enhanced_persistence_manager,
    ConnectionPool,
)

    await manager.initialize()
    await manager.close()

if __name__ == '__main__':
    asyncio.run(main())
