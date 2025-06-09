from __future__ import annotations

"""Tool launcher and guidance utilities."""
from typing import Any, Callable, Dict, Optional


class ToolGuide:
    """Provides before-run guidance for a tool."""

    def __init__(
        self, instructions: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.instructions = instructions
        self.metadata = metadata or {}

    def display(self) -> None:
        """Display guidance to the user before running a tool."""
        print("\n=== Tool Instructions ===")
        print(self.instructions)
        if self.metadata:
            print("--- Metadata ---")
            for key, value in self.metadata.items():
                print(f"{key}: {value}")
        print("========================\n")


class ToolLauncher:
    """Registers tools and provides a simple run interface with guidance."""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._guides: Dict[str, ToolGuide] = {}

    def register(
        self, name: str, func: Callable[..., Any], guide: Optional[ToolGuide] = None
    ) -> None:
        self._tools[name] = func
        if guide:
            self._guides[name] = guide

    def run_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' is not registered")
        guide = self._guides.get(name)
        if guide:
            guide.display()
        return self._tools[name](*args, **kwargs)


# Global launcher instance
default_launcher = ToolLauncher()


def register_tool(
    name: str, func: Callable[..., Any], guide: Optional[ToolGuide] = None
) -> None:
    """Register a tool with the global launcher."""
    default_launcher.register(name, func, guide)


def run_tool(name: str, *args: Any, **kwargs: Any) -> Any:
    """Run a registered tool by name."""
    return default_launcher.run_tool(name, *args, **kwargs)
