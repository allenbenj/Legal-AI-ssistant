from __future__ import annotations

"""Streamlit panel for managing memory entries."""
from typing import Any, Dict, List

import streamlit as st

from ...tools.contradiction_detector import MemoryEntry
from ...tools import run_tool, register_tool, ToolGuide
from ..memory_brain_core import (
    MemoryBrainCore,
    run_contradiction_check,
)


class MemoryBrainPanel:
    """Memory Brain panel with multiple tabs."""

    def __init__(self) -> None:
        self.core = MemoryBrainCore()
        self.core.load_entries()
        self.memory_entries = self.core.memory_entries
        self._register_tools()



    def _register_tools(self) -> None:
        """Register builtin tools used by this panel."""
        guide = ToolGuide(
            "Checks the given statement against stored memories for contradictions.",
            {"author": "MemoryBrain", "version": "1.0"},
        )

        def _contradiction_tool(
            speaker: str, statement: str, source: str
        ) -> Dict[str, Any]:
            return run_contradiction_check(
                self.memory_entries, speaker, statement, source
            )

        register_tool("contradiction_check", _contradiction_tool, guide)


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
                self.core.add_statement(entry)
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
