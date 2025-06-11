# legal_ai_system/__main__.py
#!/usr/bin/env python3
"""
Legal AI System - Module Entry Point
====================================
Run with: ``python -m legal_ai_system``
This will typically launch the user interface located at
``legal_ai_system/gui/streamlit_app.py``.
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# Ensure the package root is in sys.path if running with `python legal_ai_system/__main__.py`
# This is usually not needed if running with `python -m legal_ai_system` from one level up.
# However, adding it can make direct script execution more robust.
PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(
        0, str(PACKAGE_ROOT.parent)
    )  # Add the directory containing 'legal_ai_system'

# Configure basic logging so informational messages surface when running
# ``python -m legal_ai_system``. This keeps output consistent across
# interfaces while allowing more advanced logging setups to override if
# needed.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)


def run_streamlit_gui() -> int:
    """Launch the Streamlit dashboard."""
    try:
        from legal_ai_system.gui.streamlit_app import main_streamlit_entry

        logging.info(
            "Launching Legal AI System GUI via: legal_ai_system.gui.streamlit_app.main_streamlit_entry()"
        )
        main_streamlit_entry()
        return 0
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", str(e))
        logging.error(
            "Missing dependency '%s' required for the Streamlit GUI.",
            missing,
        )
        logging.error("Install required packages with: pip install -r requirements.txt")
        return 1
    except Exception as e:  # noqa: PIE786 - show any unexpected exception
        logging.exception("Failed to launch Streamlit GUI: %s", e)
        return 1


def run_pyqt_gui() -> int:
    """Launch the optional PyQt demonstration window."""
    try:
        from legal_ai_system.gui.main_gui import main as qt_main

        logging.info("Launching PyQt GUI via: legal_ai_system.gui.main_gui.main()")
        qt_main()
        return 0
    except Exception as e:  # noqa: PIE786
        logging.exception("Failed to launch PyQt GUI: %s", e)
        return 1


def run_api_server() -> int:
    """Start the FastAPI backend using uvicorn."""
    try:
        import uvicorn
        from legal_ai_system.scripts.main import app

        host = os.getenv("LEGAL_AI_API_HOST", "0.0.0.0")
        port = int(os.getenv("LEGAL_AI_API_PORT", "8000"))
        logging.info("Starting FastAPI server on %s:%s", host, port)
        uvicorn.run(app, host=host, port=port, reload=True)
        return 0
    except Exception as e:  # noqa: PIE786
        logging.exception("Failed to start FastAPI server: %s", e)
        return 1


def main() -> int:
    """Parse command line arguments and launch the requested mode."""
    parser = argparse.ArgumentParser(description="Launch the Legal AI System")
    parser.add_argument(
        "--mode",
        choices=["streamlit", "qt", "api"],
        default="streamlit",
        help="Interface to start: streamlit (default), qt, or api",
    )

    args = parser.parse_args()

    if args.mode == "qt":
        return run_pyqt_gui()
    if args.mode == "api":
        return run_api_server()
    return run_streamlit_gui()


if __name__ == "__main__":
    # This block is executed when the package is run directly.
    sys.exit(main())
