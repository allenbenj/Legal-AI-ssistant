import pytest

from legal_ai_system.core.enhanced_persistence import (
    EnhancedPersistenceManager,
    ConnectionPool,
)


class DummyMetrics:
    pass


def test_init_with_provided_pool():
    pool = ConnectionPool("postgresql://user@localhost/db", None)
    manager = EnhancedPersistenceManager(
        config={"cache_default_ttl_seconds": 42},
        connection_pool=pool,
        metrics_exporter=DummyMetrics(),
    )
    assert manager.connection_pool is pool
    assert manager.cache_manager.default_ttl == 42
    assert isinstance(manager.metrics, DummyMetrics)


def test_init_creates_pool():
    cfg = {
        "database_url": "postgresql://user@localhost/db",
        "redis_url": "redis://localhost/0",
        "cache_default_ttl_seconds": 10,
    }
    manager = EnhancedPersistenceManager(config=cfg)
    assert isinstance(manager.connection_pool, ConnectionPool)
    assert manager.connection_pool.database_url == cfg["database_url"]
    assert manager.connection_pool.redis_url == cfg["redis_url"]
    assert manager.cache_manager.default_ttl == 10
