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

try:  # pragma: no cover - optional dependency at runtime
    from ..utils.reviewable_memory import ReviewableMemory
except Exception:  # pragma: no cover - during tests
    ReviewableMemory = Any  # type: ignore

try:  # pragma: no cover - optional dependency
    from ..api.websocket_manager import ConnectionManager
except Exception:  # pragma: no cover - during tests
    ConnectionManager = Any  # type: ignore


def build_graph(
    topic: str,
    *,
    connection_manager: ConnectionManager | None = None,
    review_memory: ReviewableMemory | None = None,
) -> StateGraph:
    """Build a simple LangGraph pipeline for a given topic."""

    graph = StateGraph()

    graph.add_node("analysis", AnalysisNode(topic))
    if connection_manager:
        graph.add_node("progress", ProgressTrackingNode(connection_manager))
    if review_memory:
        graph.add_node("human_review", HumanReviewNode(review_memory))
    graph.add_node("summary", SummaryNode())

    graph.set_entry_point("analysis")

    if connection_manager and review_memory:
        graph.add_edge("analysis", "progress")
        graph.add_edge("progress", "human_review")
        graph.add_edge("human_review", "summary")
    elif connection_manager:
        graph.add_edge("analysis", "progress")
        graph.add_edge("progress", "summary")
    elif review_memory:
        graph.add_edge("analysis", "human_review")
        graph.add_edge("human_review", "summary")
    else:
        graph.add_edge("analysis", "summary")

    graph.add_edge("summary", END)

    return graph


__all__ = ["build_graph"]
