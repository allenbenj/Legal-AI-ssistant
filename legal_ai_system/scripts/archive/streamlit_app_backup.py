"""Legacy entry point for the Streamlit GUI.

This file previously contained an experimental version of the Streamlit
application. To keep the archive lightweight and avoid stale code errors,
it now delegates to the maintained GUI module under ``legal_ai_system.gui``.
"""

from legal_ai_system.gui.streamlit_app import main_streamlit_entry

if __name__ == "__main__":
    main_streamlit_entry()
