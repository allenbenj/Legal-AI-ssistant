"""Setup a basic LangGraph document processing workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - optional dependency
    from langgraph.graph import StateGraph, END
except Exception:  # ImportError or other issues if langgraph not installed

    class StateGraph:  # pragma: no cover - simple placeholder
        """Fallback ``StateGraph`` when ``langgraph`` is unavailable."""

        def __init__(self) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        # provide stub methods so static analysis tools see expected attributes
        def add_node(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        def set_entry_point(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        def add_edge(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

        def run(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("LangGraph is required to build workflows")

    END = "END"

if TYPE_CHECKING:  # pragma: no cover - hint for type checkers
    from langgraph.graph import StateGraph as _RealStateGraph
    from langgraph.graph import END as _RealEND

from ..agents.agent_nodes import AnalysisNode, SummaryNode
from .nodes import HumanReviewNode, ProgressTrackingNode
from ..utils.reviewable_memory import ReviewableMemory
from ..api.websocket_manager import ConnectionManager


def build_graph(topic: str) -> StateGraph:
    """Build a simple LangGraph pipeline for a given topic."""
    graph = StateGraph()

    graph.add_node("analysis", AnalysisNode(topic))
    review_memory = ReviewableMemory()
    graph.add_node("human_review", HumanReviewNode(review_memory))
    manager = ConnectionManager()
    graph.add_node("summary", SummaryNode())
    graph.add_node("progress", ProgressTrackingNode(manager))

    graph.set_entry_point("analysis")
    graph.add_edge("analysis", "human_review")
    graph.add_edge("human_review", "summary")
    graph.add_edge("summary", "progress")
    graph.add_edge("progress", END)

    return graph


__all__ = ["build_graph"]
