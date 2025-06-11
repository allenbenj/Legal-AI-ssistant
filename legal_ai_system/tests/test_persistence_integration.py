import os
import asyncio
import pytest

from legal_ai_system.core.enhanced_persistence import (
    create_enhanced_persistence_manager,
    ConnectionPool,
    EntityRecord,
    EntityStatus,
)

@pytest.mark.asyncio
async def test_entity_create_and_fetch():
    dsn = os.getenv("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("TEST_DATABASE_URL not set")

    pool = ConnectionPool(dsn, None)
    manager = create_enhanced_persistence_manager(
        None,
        config={"database_url": dsn},
        connection_pool=pool,
    )
    await manager.initialize()

    record = EntityRecord(
        entity_id="test-e1",
        entity_type="person",
        canonical_name="John Doe",
        created_by="test",
        updated_by="test",
        status=EntityStatus.ACTIVE,
    )

    await manager.entity_repo.create_entity(record)
    fetched = await manager.entity_repo.get_entity("test-e1")
    await manager.close()

    assert fetched is not None
    assert fetched.canonical_name == "John Doe"
