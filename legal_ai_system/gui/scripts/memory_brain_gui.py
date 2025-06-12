"""Standalone GUI to demonstrate the MemoryBrainPanel."""

import streamlit as st

from ..panels.memory_brain_panel import MemoryBrainPanel


def main() -> None:
    st.set_page_config(page_title="Memory Brain", page_icon="🧠", layout="wide")
    panel = MemoryBrainPanel()
    panel.render()


if __name__ == "__main__":
    main()
