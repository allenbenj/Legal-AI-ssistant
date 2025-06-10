from __future__ import annotations

from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    from langgraph.graph import BaseNode
except Exception:  # pragma: no cover - fallback when langgraph isn't installed
    class BaseNode:
        """Minimal stand-in for :class:`langgraph.graph.BaseNode`."""

        pass

from ...utils.document_utils import LegalDocumentClassifier


class DocumentClassificationNode(BaseNode):
    """Classify document text using :class:`LegalDocumentClassifier`."""

    def __init__(self, classifier: LegalDocumentClassifier | None = None) -> None:
        self.classifier = classifier or LegalDocumentClassifier()

    def __call__(self, text: str) -> Dict[str, Any]:
        """Return classification details for ``text``."""
        return self.classifier.classify(text)


__all__ = ["DocumentClassificationNode"]
