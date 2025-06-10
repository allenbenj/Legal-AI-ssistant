#!/usr/bin/env python3
"""Run health checks to verify system linkage and dependencies."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable


def run(cmd: Iterable[str]) -> None:
    """Execute a command and abort on failure."""
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    subprocess.run(list(cmd), check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    test_dir = repo_root / "legal_ai_system" / "tests"
    run([sys.executable, "-m", "pytest", str(test_dir)])


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
