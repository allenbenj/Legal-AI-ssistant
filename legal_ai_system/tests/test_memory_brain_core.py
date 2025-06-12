from contextlib import asynccontextmanager
import sys
import types

import pytest

sys.modules.setdefault("streamlit", types.SimpleNamespace())
@asynccontextmanager
async def _dummy_cm():
    yield None

sys.modules.setdefault(
    "legal_ai_system.services.memory_service",
    types.SimpleNamespace(memory_manager_context=_dummy_cm),
)
from legal_ai_system.tools.contradiction_detector import MemoryEntry


@asynccontextmanager
async def _fake_context(entries=None, store_cb=None):
    class DummyManager:
        async def get_context_window(self, session_id):
            return entries or []

        async def add_context_window_entry(self, **kwargs):
            if store_cb:
                store_cb(kwargs)

    yield DummyManager()


def test_load_memory_entries(mocker):
    entries = [
        {"entry_type": "statement", "content": "Rain", "metadata": {"speaker": "Bob", "source": "a"}},
        {"entry_type": "other", "content": "foo"},
    ]

    import importlib
    module = importlib.import_module("legal_ai_system.gui.memory_brain_core")
    mocker.patch(
        "legal_ai_system.gui.memory_brain_core.memory_manager_context",
        lambda: _fake_context(entries=entries),
    )
    result = module.load_memory_entries()
    assert result == [MemoryEntry(speaker="Bob", statement="Rain", source="a")]


def test_persist_statement(mocker):
    recorded = {}
    def store_cb(kwargs):
        recorded.update(kwargs)

    import importlib
    module = importlib.import_module("legal_ai_system.gui.memory_brain_core")
    mocker.patch(
        "legal_ai_system.gui.memory_brain_core.memory_manager_context",
        lambda: _fake_context(store_cb=store_cb),
    )
    entry = MemoryEntry(speaker="Ann", statement="Hi", source="b")
    module.persist_statement(entry)
    assert recorded["content"] == "Hi"
    assert recorded["metadata"] == {"speaker": "Ann", "source": "b"}


def test_memory_brain_core_add_and_check(mocker):
    import importlib
    module = importlib.import_module("legal_ai_system.gui.memory_brain_core")
    core = module.MemoryBrainCore()
    mocker.patch(
        "legal_ai_system.gui.memory_brain_core.persist_statement",
        lambda e: None,
    )
    core.add_statement(MemoryEntry(speaker="Bob", statement="It is raining", source="x"))
    res = core.check("Bob", "It is not raining", "y")
    assert res["count"] == 1


def test_run_contradiction_check():
    import importlib
    module = importlib.import_module("legal_ai_system.gui.memory_brain_core")
    entries = [MemoryEntry(speaker="Bob", statement="yes", source="s")]
    result = module.run_contradiction_check(entries, "Bob", "not yes", "x")
    assert result["count"] == 1
