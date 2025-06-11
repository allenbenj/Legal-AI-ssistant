import asyncio
import sqlite3
from pathlib import Path

import pytest

from legal_ai_system.core.unified_memory_manager import UnifiedMemoryManager, MemoryType, MemoryEntry

@pytest.mark.asyncio
async def test_agent_memory_type(tmp_path):
    db_path = tmp_path / "umm.db"
    umm = UnifiedMemoryManager(db_path_str=str(db_path))
    await umm.initialize()

    entry = await umm.store_agent_memory(
        session_id="s1",
        agent_name="agent",
        key="foo",
        value={"bar": 1},
        memory_type=MemoryType.AGENT_SPECIFIC,
    )

    assert isinstance(entry, MemoryEntry)
    assert entry.memory_type == MemoryType.AGENT_SPECIFIC

    retrieved = await umm.retrieve_agent_memory(
        session_id="s1",
        agent_name="agent",
        key="foo",
        memory_type=MemoryType.AGENT_SPECIFIC,
    )
    assert isinstance(retrieved, MemoryEntry)
    assert retrieved.memory_type == MemoryType.AGENT_SPECIFIC
    assert retrieved.value == {"bar": 1}

@pytest.mark.asyncio
async def test_context_window_memory_type(tmp_path):
    db_path = tmp_path / "umm.db"
    umm = UnifiedMemoryManager(db_path_str=str(db_path))
    await umm.initialize()

    entry = await umm.add_context_window_entry(
        session_id="s1",
        entry_type="user",
        content="hello",
        memory_type=MemoryType.CONTEXT_WINDOW,
    )

    assert entry.memory_type == MemoryType.CONTEXT_WINDOW

    entries = await umm.get_context_window("s1")
    assert len(entries) == 1
    assert entries[0].memory_type == MemoryType.CONTEXT_WINDOW
    assert entries[0].value == "hello"
