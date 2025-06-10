#!/usr/bin/env python3
"""Automate environment setup and verification for the Legal AI System."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def run(cmd: Iterable[str], timeout: int = 600) -> None:
    """Run a shell command and handle errors/timeout."""
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"Timeout executing: {' '.join(cmd)}")
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}")


def ensure_venv(venv_path: Path) -> None:
    """Create a Python virtual environment if it does not exist."""
    if not venv_path.exists():
        run([sys.executable, "-m", "venv", str(venv_path)])


def pip(venv_path: Path, args: Iterable[str]) -> None:
    """Execute pip inside the virtual environment."""
    pip_exe = venv_path / "bin" / "pip"
    run([str(pip_exe), *args])


def verify_imports(venv_path: Path) -> None:
    """Check that critical packages can be imported."""
    python_exe = venv_path / "bin" / "python"
    code = (
        "import fastapi, uvicorn, streamlit, pydantic;"
        "import openai, sentence_transformers;"
        "import neo4j, sqlalchemy, lancedb;"
        "import asyncpg, aioredis, torch;"
        "print('Environment ready')"
    )
    run([str(python_exe), "-c", code])


def run_tests(venv_path: Path) -> None:
    """Run the project's pytest suite."""
    python_exe = venv_path / "bin" / "python"
    run([str(python_exe), "-m", "pytest"])


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    venv_path = repo_root / ".venv"
    ensure_venv(venv_path)

    pip(venv_path, ["install", "--upgrade", "pip"])
    pip(venv_path, ["install", "-r", str(repo_root / "requirements.txt")])

    verify_imports(venv_path)
    run_tests(venv_path)


if __name__ == "__main__":
    main()

