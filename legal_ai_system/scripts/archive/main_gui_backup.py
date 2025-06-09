"""Deprecated GUI entry point.

This module originally contained a larger Streamlit application. The
active implementation now resides in :mod:`legal_ai_system.gui.streamlit_app`.
This file is kept for historical reference and simply launches the current
version when executed.
"""

from legal_ai_system.gui.streamlit_app import main_streamlit_entry


def main() -> None:
    """Launch the maintained Streamlit GUI."""
    main_streamlit_entry()


if __name__ == "__main__":
    main()
