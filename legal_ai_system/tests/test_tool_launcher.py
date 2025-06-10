import pytest
from legal_ai_system.tools.tool_launcher import ToolLauncher, register_tool, run_tool, default_launcher


def test_tool_launcher_register_and_run():
    launcher = ToolLauncher()

    def hello(name: str) -> str:
        return f"hello {name}"

    launcher.register("hello", hello)
    assert launcher.run_tool("hello", "world") == "hello world"


def test_tool_launcher_missing_tool():
    launcher = ToolLauncher()
    with pytest.raises(ValueError):
        launcher.run_tool("missing")


def test_global_register_and_run():
    def add(a: int, b: int) -> int:
        return a + b
    register_tool("add", add)
    try:
        assert run_tool("add", 1, 2) == 3
    finally:
        # cleanup global launcher
        default_launcher._tools.pop("add", None)
        default_launcher._guides.pop("add", None)
