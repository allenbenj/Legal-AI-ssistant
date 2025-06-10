import asyncio

from langgraph.graph import StateGraph, END


def test_stategraph_conditional_routing():
    graph = StateGraph()
    graph.add_node("start", lambda x: x)
    graph.add_node("pos", lambda x: x + 1)
    graph.add_node("neg", lambda x: x - 1)

    graph.set_entry_point("start")
    graph.add_conditional_edges(
        "start", [(lambda v: v > 0, "pos"), (lambda v: v <= 0, "neg")]
    )
    graph.add_edge("pos", END)
    graph.add_edge("neg", END)

    assert graph.run(2) == 3
    assert graph.run(-1) == -2


def test_stategraph_parallel_merge():
    graph = StateGraph()
    graph.add_node("start", lambda x: x)

    async def node_a(x):
        await asyncio.sleep(0.01)
        return x + 1

    async def node_b(x):
        await asyncio.sleep(0.01)
        return x + 2

    def merge(results):
        return sum(results)

    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.add_node("merge", merge)

    graph.set_entry_point("start")
    graph.add_parallel_nodes("start", ["a", "b"], "merge")
    graph.add_edge("merge", END)

    assert graph.run(0) == 3

