"""Command-line interface to run registered tools."""

from pathlib import Path
from typing import List

import typer

# Ensure package is importable when executed directly
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from legal_ai_system.tools import run_tool


app = typer.Typer(help="Execute any registered Legal AI System tool")


@app.command()
def run(
    name: str, args: List[str] = typer.Argument([], help="Arguments for the tool")
) -> None:
    """Run the specified tool with optional positional arguments."""
    result = run_tool(name, *args)
    typer.echo(result)


def main() -> None:  # pragma: no cover - thin wrapper for Typer
    app()


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
