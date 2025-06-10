from __future__ import annotations

try:
    from langgraph.graph import BaseNode
except Exception:  # pragma: no cover - optional dependency
    class BaseNode:  # type: ignore[misc]
        """Fallback ``BaseNode`` when LangGraph is unavailable."""
        pass


class DocumentClassificationNode(BaseNode):
    """Simple rule-based classifier for legal documents."""

    def __call__(self, document: str) -> str:
        text = document.lower()
        if "contract" in text:
            return "contract"
        if "litigation" in text or "lawsuit" in text:
            return "litigation"
        if "regulation" in text:
            return "regulatory"
        if "evidence" in text:
            return "evidence"
        return "unknown"


__all__ = ["DocumentClassificationNode"]
