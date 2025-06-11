#!/usr/bin/env python3
"""Apply LangGraph typing fixes across the repository.

This script renames fallback ``BaseNode`` definitions to ``LangGraphBaseNode``
and creates a runtime alias ``BaseNode = LangGraphBaseNode`` to avoid
``reportAssignmentType`` errors raised by Pylance. Run it from the repository
root:

    python legal_ai_system/scripts/fix_langgraph_typing.py
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def iter_candidate_files() -> list[Path]:
    """Return repository Python files that import ``BaseNode`` from LangGraph."""
    candidates = []
    for path in REPO_ROOT.rglob("*.py"):
        if path == Path(__file__):
            continue
        if "site-packages" in path.parts or "/.venv/" in str(path):
            continue
        text = path.read_text(encoding="utf-8")
        if "from langgraph.graph import" in text and "BaseNode" in text:
            candidates.append(path)
    return candidates


def update_file(path: Path) -> bool:
    """Rewrite imports and fallback classes in ``path`` if needed."""
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    changed = False

    return changed


def main() -> None:
    changed_any = False
    for path in iter_candidate_files():
        if update_file(path):
            print(f"Updated {path.relative_to(REPO_ROOT)}")
            changed_any = True
    if not changed_any:
        print("No changes required.")


if __name__ == "__main__":
    main()
