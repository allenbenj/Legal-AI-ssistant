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
    alias_line = "BaseNode = LangGraphBaseNode"
    alias_idx = None
    import_pos = None
    fallback_start = None
    insert_pos = None
    for i, line in enumerate(lines):
        if "from langgraph.graph" in line and "BaseNode" in line:
            if "BaseNode as" not in line:
                lines[i] = line.replace("BaseNode", "BaseNode as LangGraphBaseNode")
                changed = True
            if import_pos is None:
                import_pos = i
        if line.lstrip().startswith("class BaseNode"):
            lines[i] = line.replace("class BaseNode", "class LangGraphBaseNode", 1)
            changed = True
            fallback_start = i
        elif line.lstrip().startswith("class LangGraphBaseNode"):
            fallback_start = i
        if fallback_start is not None and insert_pos is None and i > fallback_start and line.lstrip().startswith("pass"):
            insert_pos = i + 1
        if line.strip() == alias_line:
            if alias_idx is None:
                alias_idx = i
            else:
                lines[i] = None
                changed = True
    lines = [l for l in lines if l is not None]
    if insert_pos is None:
        insert_pos = (fallback_start + 1 if fallback_start is not None else
                      (import_pos + 1 if import_pos is not None else len(lines)))
    if alias_idx is None:
        indent = ""
        if insert_pos > 0:
            prev = lines[insert_pos - 1]
            indent = prev[: len(prev) - len(prev.lstrip())]
        lines.insert(insert_pos, indent + alias_line)
        changed = True
    elif alias_idx != insert_pos:
        line = lines.pop(alias_idx)
        if insert_pos > len(lines):
            insert_pos = len(lines)
        lines.insert(insert_pos, line)
        changed = True
    if changed:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
