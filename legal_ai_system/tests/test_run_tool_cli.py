import importlib
from typer.testing import CliRunner
import pytest

from legal_ai_system.tools.tool_launcher import register_tool, default_launcher


runner = CliRunner()


def test_run_cli_executes_registered_tool():
    def greet(name: str) -> str:
        return f"hi {name}"

    register_tool("greet", greet)
    try:
        module = importlib.import_module("legal_ai_system.scripts.run_tool_cli")
        result = runner.invoke(module.app, ["greet", "Alice"])
        assert result.exit_code == 0
        assert "hi Alice" in result.stdout
    finally:
        default_launcher._tools.pop("greet", None)
        default_launcher._guides.pop("greet", None)


def test_run_cli_unregistered_tool():
    module = importlib.import_module("legal_ai_system.scripts.run_tool_cli")
    result = runner.invoke(module.app, ["missing"])
    assert result.exit_code != 0
    assert "not registered" in str(result.exception)
