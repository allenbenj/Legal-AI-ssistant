import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib
import types

# Expose the local LangGraph stub when the real package is missing
if importlib.util.find_spec("langgraph") is None:
    mod = importlib.import_module("legal_ai_system.langgraph")
    sys.modules.setdefault("langgraph", mod)
    sys.modules.setdefault(
        "langgraph.graph", importlib.import_module("legal_ai_system.langgraph.graph")
    )


