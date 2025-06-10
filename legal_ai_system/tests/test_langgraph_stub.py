import asyncio

from langgraph.graph import StateGraph, END


def test_conditional_routing() -> None:
    graph = StateGraph()
    graph.add_node("start", lambda x: x)
    graph.add_node("pos", lambda x: x + 1)
    graph.add_node("neg", lambda x: x - 1)
    graph.add_node("final", lambda x: x)

    graph.set_entry_point("start")
    graph.add_conditional_edges(
        "start",
        [
            (lambda v: v > 0, "pos"),
            (lambda v: v <= 0, "neg"),
        ],
    )
    graph.add_edge("pos", "final")
    graph.add_edge("neg", "final")
    graph.add_edge("final", END)

    assert graph.run(5) == 6
    assert graph.run(-1) == -2


def test_parallel_merge() -> None:
    graph = StateGraph()
    calls = {"a": 0, "b": 0}

    graph.add_node("start", lambda x: x)

    async def node_a(x: int) -> int:
        calls["a"] += 1
        await asyncio.sleep(0.01)
        return x + 1

    async def node_b(x: int) -> int:
        calls["b"] += 1
        await asyncio.sleep(0.01)
        return x + 2

    async def merge(results: list[int]) -> int:
        return sum(results)

    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.add_node("merge", merge)

    graph.set_entry_point("start")
    graph.add_parallel_nodes("start", ["a", "b"], "merge")
    graph.add_edge("merge", END)

    assert graph.run(0) == 3
    assert calls == {"a": 1, "b": 1}
