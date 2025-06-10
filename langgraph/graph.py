


class BaseNode:
    """Minimal ``BaseNode`` placeholder used for local development."""

    pass


class StateGraph:
    """Simplified asynchronous graph structure used for workflows."""

    def __init__(self) -> None:


        self._nodes[name] = node

    def set_entry_point(self, name: str) -> None:
        self._entry_point = name



        if self._entry_point is None:
            raise RuntimeError("Entry point not set")


END = "END"
