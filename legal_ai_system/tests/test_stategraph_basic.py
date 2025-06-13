import unittest

from langgraph.graph import StateGraph, END


class TestStateGraphBasic(unittest.TestCase):
    """Verify core StateGraph functionality."""

    def test_single_node(self) -> None:
        graph = StateGraph()
        graph.add_node("start", lambda v: v + 1)
        graph.set_entry_point("start")
        graph.add_edge("start", END)

        self.assertEqual(graph.run(1), 2)

    def test_edge_execution(self) -> None:
        graph = StateGraph()
        graph.add_node("a", lambda v: v + 1)
        graph.add_node("b", lambda v: v * 2)

        graph.set_entry_point("a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)

        self.assertEqual(graph.run(1), 4)

    def test_conditional_routing(self) -> None:
        graph = StateGraph()
        graph.add_node("start", lambda v: v)
        graph.add_node("pos", lambda v: v + 1)
        graph.add_node("neg", lambda v: v - 1)

        graph.set_entry_point("start")
        graph.add_conditional_edges(
            "start",
            [
                (lambda v: v >= 0, "pos"),
                (lambda v: v < 0, "neg"),
            ],
        )
        graph.add_edge("pos", END)
        graph.add_edge("neg", END)

        self.assertEqual(graph.run(2), 3)
        self.assertEqual(graph.run(-3), -4)


if __name__ == "__main__":  # pragma: no cover - manual run
    unittest.main()

