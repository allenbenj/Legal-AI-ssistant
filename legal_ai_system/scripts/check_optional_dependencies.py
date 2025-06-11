#!/usr/bin/env python3
"""Verify optional dependencies used by advanced processing agents."""
from __future__ import annotations

import importlib
import sys
from typing import Iterable

OPTIONAL_MODULES: Iterable[str] = [
    "moviepy.editor",
    "whisper",
    "pdfplumber",
    "openpyxl",
    "docx",
    "lxml",
]


def main() -> None:
    missing = []
    for module in OPTIONAL_MODULES:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(module)

    if missing:
        mods = ", ".join(missing)
        sys.stderr.write(
            f"Missing optional dependencies: {mods}.\n"
            "Run `python legal_ai_system/scripts/install_all_dependencies.py` to install them.\n"
        )
        raise SystemExit(1)
    print("All optional dependencies present.")


if __name__ == "__main__":
    main()
