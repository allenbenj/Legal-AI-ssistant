from contextlib import asynccontextmanager

import pytest
import fakeredis.aioredis

import sys
from types import ModuleType

pyd = ModuleType("pydantic")
pyd.BaseModel = object
pyd.BaseSettings = object
def _field(default=None, **kwargs):
    return default
pyd.Field = _field
sys.modules.setdefault("pydantic", pyd)
if "asyncpg" not in sys.modules:
    asyncpg_mod = ModuleType("asyncpg")
    asyncpg_mod.Connection = object
    sys.modules["asyncpg"] = asyncpg_mod
if "aioredis" not in sys.modules:
    aioredis_mod = ModuleType("aioredis")
    exc = ModuleType("exceptions")
    exc.RedisError = type("RedisError", (Exception,), {})
    aioredis_mod.Redis = object
    aioredis_mod.exceptions = exc
    sys.modules["aioredis"] = aioredis_mod
if "aiofiles" not in sys.modules:
    aiofiles_mod = ModuleType("aiofiles")
    class _AsyncFile:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def read(self):
            return "{}"
        async def write(self, _):
            pass
    async def open(*a, **k):
        return _AsyncFile()
    aiofiles_mod.open = open
    sys.modules["aiofiles"] = aiofiles_mod

from legal_ai_system.core.enhanced_persistence import CacheManager

class FakePool:
    def __init__(self):
        self.redis = fakeredis.aioredis.FakeRedis()
        self.redis_pool = self.redis

    @asynccontextmanager
    async def get_redis_connection(self):
        yield self.redis

@pytest.mark.asyncio
async def test_cache_persist_and_load(tmp_path):
    pool = FakePool()
    cm = CacheManager(pool, default_ttl_seconds=60)
    await cm.set("alpha", {"x": 1})
    cache_file = tmp_path / "cache.json"
    saved = await cm.persist_to_disk(str(cache_file))
    assert saved is True
    await cm.delete("alpha")
    assert await cm.get("alpha") is None
    loaded = await cm.load_from_disk(str(cache_file))
    assert loaded == 1
    value = await cm.get("alpha")
    assert value == {"x": 1}
