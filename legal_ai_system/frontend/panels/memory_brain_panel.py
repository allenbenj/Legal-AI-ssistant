from __future__ import annotations

"""Streamlit panel for managing memory entries."""
from typing import Any, Dict, List
import asyncio

import streamlit as st

from ...tools.contradiction_detector import ContradictionDetector, MemoryEntry
from ...tools import run_tool, register_tool, ToolGuide
from ...services.memory_service import memory_manager_context


class MemoryBrainPanel:
    """Memory Brain panel with multiple tabs."""

    def __init__(self) -> None:
        self.memory_entries: List[MemoryEntry] = []
        self._load_memory_entries()
        self._register_tools()

    def _load_memory_entries(self) -> None:
        """Load memory entries from the :class:`UnifiedMemoryManager`."""

        async def _load() -> None:
            # Retrieve previously stored statements from UnifiedMemoryManager
            async with memory_manager_context() as manager:
                entries = await manager.get_context_window("memory_brain")
                self.memory_entries = [
                    MemoryEntry(
                        speaker=e.get("metadata", {}).get("speaker", ""),
                        statement=e.get("content", ""),
                        source=e.get("metadata", {}).get("source", ""),
                    )
                    for e in entries
                    if e.get("entry_type") == "statement"
                ]

        try:
            asyncio.run(_load())
        except Exception:
            self.memory_entries = []

    def _register_tools(self) -> None:
        """Register builtin tools used by this panel."""
        guide = ToolGuide(
            "Checks the given statement against stored memories for contradictions.",
            {"author": "MemoryBrain", "version": "1.0"},
        )

        def _contradiction_tool(
            speaker: str, statement: str, source: str
        ) -> Dict[str, Any]:
            detector = ContradictionDetector(self.memory_entries)
            return detector.check(speaker, statement, source)

        register_tool("contradiction_check", _contradiction_tool, guide)

    def _persist_statement(self, entry: MemoryEntry) -> None:
        """Persist a statement via :class:`UnifiedMemoryManager`."""

        async def _store() -> None:
            async with memory_manager_context() as manager:
                await manager.add_context_window_entry(
                    session_id="memory_brain",
                    entry_type="statement",
                    content=entry.statement,
                    metadata={"speaker": entry.speaker, "source": entry.source},
                )

        try:
            asyncio.run(_store())
        except Exception:
            pass

    def render(self) -> None:
        """Render the Memory Brain panel."""
        tabs = st.tabs(["Statement Intake", "Contradiction Check", "Merge & Curate"])

        with tabs[0]:
            st.subheader("Statement Intake")
            speaker = st.text_input("Speaker")
            statement = st.text_area("Statement")
            source = st.text_input("Source")
            if st.button("Add Statement"):
                entry = MemoryEntry(speaker=speaker, statement=statement, source=source)
                self.memory_entries.append(entry)
                self._persist_statement(entry)
                st.success("Statement added to memory.")

        with tabs[1]:
            st.subheader("Contradiction Check")
            cspeaker = st.text_input("Speaker", key="cspeaker")
            cstatement = st.text_area("Statement", key="cstatement")
            csource = st.text_input("Source", key="csource")
            if st.button("Run Check"):
                result = run_tool("contradiction_check", cspeaker, cstatement, csource)
                st.info(f"Contradictions found: {result['count']}")
                if result["contradictions"]:
                    st.json(result["contradictions"])

        with tabs[2]:
            st.subheader("Merge & Curate")
            st.info("Additional curation tools would appear here.")
