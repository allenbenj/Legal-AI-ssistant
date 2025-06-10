import asyncio


class BaseNode:
    """Minimal BaseNode placeholder for local development."""

    pass


class StateGraph:
    """Simplified graph structure used for workflows."""

    def __init__(self) -> None:
        self._nodes = {}
        # edge: (start, end, condition, parallel)
        self._edges = []
        self._entry_point = None
        # mapping from start node to (branches, merge)
        self._parallel_groups = {}

    def add_node(self, name, node):
        self._nodes[name] = node

    def set_entry_point(self, name):
        self._entry_point = name

    def add_edge(self, start, end, condition=None, parallel=False):
        self._edges.append((start, end, condition, parallel))

    def add_parallel_nodes(self, start, branches, merge):
        self._parallel_groups[start] = (list(branches), merge)
        for branch in branches:
            self.add_edge(start, branch, parallel=True)

    def add_conditional_edges(self, start, edges):
        for condition, dest in edges:
            self.add_edge(start, dest, condition=condition)

    def run(self, input_text):
        if self._entry_point is None:
            raise RuntimeError("Entry point not set")
        return asyncio.run(self._execute(self._entry_point, input_text))

    async def _call_node(self, name, data):
        func = self._nodes[name]
        result = func(data)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def _execute(self, current, data):
        if current == END:
            return data

        result = await self._call_node(current, data)

        if current in self._parallel_groups:
            branches, merge_node = self._parallel_groups[current]
            branch_results = await asyncio.gather(
                *(self._call_node(b, result) for b in branches)
            )
            result = await self._call_node(merge_node, branch_results)
            current = merge_node

        edges = [e for e in self._edges if e[0] == current]
        if not edges:
            if current == END:
                return result
            raise RuntimeError(f"No edge from {current}")

        for _, dest, cond, _ in edges:
            if cond is None or cond(result):
                return await self._execute(dest, result)
        # If no conditions matched, remain at current
        return result


END = "END"
