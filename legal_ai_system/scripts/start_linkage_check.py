#!/usr/bin/env python3
"""Run system health checks and the test suite."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    """Execute a command and fail if it errors."""
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    python_exe = sys.executable
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from legal_ai_system.services.system_initializer import test_system_health

        health = test_system_health()
        print("System health summary:")
        for key, value in health.items():
            print(f"  {key}: {value}")
    except Exception as exc:  # pragma: no cover - best effort
        print(f"Health check failed: {exc}")

    run([python_exe, "-m", "nose2", str(repo_root / "legal_ai_system" / "tests")])


if __name__ == "__main__":
    main()
