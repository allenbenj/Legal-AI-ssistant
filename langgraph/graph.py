class BaseNode:
    """Minimal BaseNode placeholder for local development."""

    pass


class StateGraph:
    """Simplified graph structure used for workflows."""

    def __init__(self) -> None:
        self._nodes = {}
        self._edges = []
        self._entry_point = None

    def add_node(self, name, node):
        self._nodes[name] = node

    def set_entry_point(self, name):
        self._entry_point = name

    def add_edge(self, start, end):
        self._edges.append((start, end))

    def run(self, input_text):
        if self._entry_point is None:
            raise RuntimeError("Entry point not set")
        current = self._entry_point
        data = input_text
        while current != END:
            node = self._nodes[current]
            data = node(data)
            next_edges = [edge for edge in self._edges if edge[0] == current]
            if not next_edges:
                raise RuntimeError(f"No edge from {current}")
            # assume single path for this stub
            current = next_edges[0][1]
        return data


END = "END"
