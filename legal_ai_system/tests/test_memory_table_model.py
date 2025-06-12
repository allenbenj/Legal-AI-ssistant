import asyncio
from pathlib import Path

import pandas as pd
import pytest

from legal_ai_system.core.unified_memory_manager import MemoryType, UnifiedMemoryManager
from legal_ai_system.gui.memory_table_model import MemoryTableModel


@pytest.mark.asyncio
async def test_memory_table_model_load(tmp_path: Path) -> None:
    db_path = tmp_path / "umm.db"
    umm = UnifiedMemoryManager(db_path_str=str(db_path))
    await umm.initialize()

    await umm.add_context_window_entry(
        session_id="s1",
        entry_type="statement",
        content="hello",
        memory_type=MemoryType.CONTEXT_WINDOW,
        metadata={"speaker": "A"},
    )
    await umm.add_context_window_entry(
        session_id="s1",
        entry_type="statement",
        content="hi",
        memory_type=MemoryType.CONTEXT_WINDOW,
        metadata={"speaker": "B"},
    )

    model = MemoryTableModel(session_id="s1", manager=umm)
    model.load()
    df = model.get_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df["speaker"]) == {"A", "B"}

    await umm.close()
