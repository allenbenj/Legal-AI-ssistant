# legal_ai_system/__main__.py
#!/usr/bin/env python3
"""
Legal AI System - Module Entry Point
====================================
Run with: ``python -m legal_ai_system``
This will typically launch the user interface located at
``legal_ai_system/gui/streamlit_app.py``.
"""


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

