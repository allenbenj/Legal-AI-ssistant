#!/usr/bin/env python3
"""Legal AI System package entry point.

Running ``python -m legal_ai_system`` launches the integrated PyQt6
application located at ``legal_ai_system/gui/legal_ai_pyqt6_enhanced.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the package root is discoverable when executed directly
PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from .gui.legal_ai_pyqt6_enhanced import main as start_gui

if __name__ == "__main__":
    start_gui()
