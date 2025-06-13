#!/usr/bin/env python3
"""Legal AI System package entry point.

Running ``python -m legal_ai_system`` launches the integrated PyQt6
application located at ``legal_ai_system/gui/legal_ai_pyqt6_integrated.py``.
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


try:
    from .gui.legal_ai_pyqt6_integrated import main as start_gui
except Exception as exc:  # pragma: no cover - runtime dependency check
    logging.error("Failed to load PyQt6 GUI: %s", exc)
    start_gui = None


if __name__ == "__main__":
    if start_gui:
        start_gui()
    else:
        # Fall back to CLI execution when PyQt6 dependencies are missing
        try:
            from .scripts.run_tool_cli import main as start_cli
        except Exception:  # pragma: no cover - final fallback
            logging.critical(
                "GUI unavailable and CLI failed to load; please check dependencies"
            )
            sys.exit(1)
        start_cli()
