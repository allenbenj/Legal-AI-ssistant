import pytest
from pathlib import Path

from legal_ai_system.core.unified_memory_manager import UnifiedMemoryManager


@pytest.mark.asyncio
async def test_shared_memory_between_agents(tmp_path: Path) -> None:
    db_path = tmp_path / "umm.db"
    mm = UnifiedMemoryManager(db_path_str=str(db_path))
    await mm.initialize()

    session_id = "sess1"
    await mm.store_shared_memory(session_id=session_id, key="item", value={"a": 1})

    value = await mm.retrieve_shared_memory(session_id=session_id, key="item")
    assert value == {"a": 1}

    await mm.close()
