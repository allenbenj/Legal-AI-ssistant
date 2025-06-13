#!/usr/bin/env python3
"""Install all Python and Node dependencies for the Legal AI System."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable


def run(
    cmd: Iterable[str], cwd: Path | None = None, timeout: int = 1800
) -> None:
    """Run a command and raise an error if it fails."""
    print(f"Running: {' '.join(cmd)}")
    # Convert to list to satisfy type checkers expecting ``Sequence[str]``
    subprocess.run(list(cmd), check=True, cwd=cwd, timeout=timeout)


def ensure_venv(venv_path: Path) -> None:
    """Create a Python virtual environment if it doesn't exist."""
    if not venv_path.exists():
        run([sys.executable, "-m", "venv", str(venv_path)])


def pip(venv_path: Path, args: Iterable[str]) -> None:
    """Execute pip within the virtual environment."""
    pip_exe = venv_path / "bin" / "pip"
    run([str(pip_exe), *args])


def npm(args: Iterable[str], cwd: Path) -> None:
    """Run npm in the provided directory."""
    run(["npm", *args], cwd=cwd)


def verify_imports(venv_path: Path) -> None:
    """Import critical libraries to confirm installation worked."""
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
    """Run nose2 using the virtual environment."""
    python_exe = venv_path / "bin" / "python"
    run([str(python_exe), "-m", "nose2"])


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    venv_path = repo_root / ".venv"

    ensure_venv(venv_path)

    # Python dependencies
    pip(venv_path, ["install", "--upgrade", "pip"])
    pip(venv_path, ["install", "packaging<25"])
    pip(venv_path, ["install", "-r", str(repo_root / "requirements.txt")])
    dev_reqs = repo_root / "requirements-dev.txt"
    if dev_reqs.exists():
        pip(venv_path, ["install", "-r", str(dev_reqs)])
    pip(venv_path, ["install", "lexnlp"])
    pip(venv_path, ["install", "langgraph", "sqlalchemy", "lancedb"])
    pip(
        venv_path,
        [
            "install",
            "ffmpeg-python",
            "moviepy",
            "openai-whisper",
            "whisperx",
            "pdfplumber",
            "openpyxl",
            "pyannote.audio",
        ],
    )

    verify_imports(venv_path)

    # Verify optional dependencies used by advanced agents
    run([
        str(venv_path / "bin" / "python"),
        str(repo_root / "legal_ai_system" / "scripts" / "check_optional_dependencies.py"),
    ])

    run_tests(venv_path)

    # Node dependencies
    root_pkg = repo_root / "package.json"
    if root_pkg.exists():
        npm(["install"], cwd=repo_root)

    frontend_dir = repo_root / "frontend"
    if (frontend_dir / "package.json").exists():
        npm(["install"], cwd=frontend_dir)

    print("All dependencies installed successfully")


if __name__ == "__main__":
    main()
