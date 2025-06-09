"""Setup a basic LangGraph document processing workflow."""

from __future__ import annotations


try:  # pragma: no cover - optional dependency
    from langgraph.graph import StateGraph, END
except Exception:  # ImportError or other issues if langgraph not installed

    class StateGraph:  # pragma: no cover - simple placeholder
        """Fallback StateGraph when ``langgraph`` is unavailable."""

        def __init__(self) -> None:
            raise RuntimeError("LangGraph is required to build workflows")

    END = "END"

from ..agents.agent_nodes import AnalysisNode, SummaryNode


def build_graph(topic: str) -> StateGraph:
    """Build a simple LangGraph pipeline for a given topic."""
    graph = StateGraph()

    graph.add_node("analysis", AnalysisNode(topic))
    graph.add_node("summary", SummaryNode())

    graph.set_entry_point("analysis")
    graph.add_edge("analysis", "summary")
    graph.add_edge("summary", END)

    return graph


__all__ = ["build_graph"]
