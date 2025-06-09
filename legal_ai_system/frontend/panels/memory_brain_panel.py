from __future__ import annotations

"""Streamlit panel for managing memory entries."""
from pathlib import Path
import json
import os
from typing import Any, Dict, List

import streamlit as st

from ...tools.contradiction_detector import ContradictionDetector, MemoryEntry
from ...tools import run_tool, register_tool, ToolGuide


class MemoryBrainPanel:
    """Memory Brain panel with multiple tabs."""

    def __init__(self) -> None:
        self.memory_entries: List[MemoryEntry] = []
        self._load_memory_entries()
        self._register_tools()

    def _load_memory_entries(self) -> None:
        """Load memory entries from JSON if available."""
        env_path = os.getenv("MEMORY_BRAIN_DATA")
        if env_path:
            data_path = Path(env_path)
        else:
            data_path = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "sample_memory_entries.json"
            )
        if data_path.exists():
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    raw_entries: List[Dict[str, Any]] = json.load(f)
                self.memory_entries = [MemoryEntry(**entry) for entry in raw_entries]
            except json.JSONDecodeError:
                self.memory_entries = []
        else:
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

    def render(self) -> None:
        """Render the Memory Brain panel."""
        tabs = st.tabs(["Statement Intake", "Contradiction Check", "Merge & Curate"])

        with tabs[0]:
            st.subheader("Statement Intake")
            speaker = st.text_input("Speaker")
            statement = st.text_area("Statement")
            source = st.text_input("Source")
            if st.button("Add Statement"):
                self.memory_entries.append(
                    MemoryEntry(speaker=speaker, statement=statement, source=source)
                )
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
