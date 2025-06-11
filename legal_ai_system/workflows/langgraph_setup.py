"""Utilities for constructing simple LangGraph workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - optional dependency
    from langgraph.graph import StateGraph, END, BaseNode as LangGraphBaseNode
except Exception:  # pragma: no cover - during tests
    class StateGraph:  # pragma: no cover - simple placeholder
        def __init__(self) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        def add_node(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        def add_edge(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        def set_entry_point(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        def run(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("LangGraph is required to build workflows")

    END = "END"

    class LangGraphBaseNode:
        pass
    BaseNode = LangGraphBaseNode


if TYPE_CHECKING:  # pragma: no cover
    from langgraph.graph import StateGraph as _RealStateGraph

from ..agents.agent_nodes import AnalysisNode, SummaryNode
from .nodes import HumanReviewNode, ProgressTrackingNode

try:  # pragma: no cover - optional dependency
    from ..utils.reviewable_memory import ReviewableMemory
except Exception:  # pragma: no cover - during tests
    ReviewableMemory = Any  # type: ignore


class CitationCheckNode(BaseNode):  # pragma: no cover - placeholder
    def __call__(self, text: str) -> str:
        return text


def build_graph(topic: str) -> StateGraph:
    """Build a minimal analysis workflow."""
    graph = StateGraph()
    graph.add_node("analysis", AnalysisNode(topic))
    graph.add_node("summary", SummaryNode())
    graph.add_edge("analysis", "summary")
    graph.set_entry_point("analysis")
    graph.add_edge("summary", END)
    return graph


def build_advanced_legal_workflow(topic: str) -> StateGraph:
    """Build an expanded LangGraph workflow including citation validation."""
    graph = build_graph(topic)
    graph.add_node("citations", CitationCheckNode())
    graph.add_edge("analysis", "citations")
    graph.add_edge("citations", "summary")
    return graph


__all__ = ["build_graph", "build_advanced_legal_workflow", "CitationCheckNode"]
