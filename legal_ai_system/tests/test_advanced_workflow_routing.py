import sys
from types import ModuleType

import pytest

# Stub heavy dependencies before importing the workflow package
if "pydantic" not in sys.modules:
    mod = ModuleType("pydantic")
    mod.BaseModel = object
    sys.modules["pydantic"] = mod

for name in [
    "legal_ai_system.services.service_container",
    "legal_ai_system.services.realtime_analysis_workflow",
    "legal_ai_system.services.workflow_orchestrator",
]:
    mod = sys.modules.setdefault(name, ModuleType(name))
    if name.endswith("service_container") and not hasattr(mod, "ServiceContainer"):
        mod.ServiceContainer = object
    if name.endswith("realtime_analysis_workflow") and not hasattr(mod, "RealTimeAnalysisWorkflow"):
        mod.RealTimeAnalysisWorkflow = object

# Minimal stub packages required for workflow modules
langgraph_mod = ModuleType("langgraph.graph")

class _Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.cond = {}
        self.entry = None

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def set_entry_point(self, name):
        self.entry = name

    def add_edge(self, src, dst):
        self.edges.setdefault(src, []).append(dst)

    def add_conditional_edges(self, src, mapping):
        self.cond[src] = mapping

    def run(self, data):
        node = self.entry
        result = data
        while node != END:
            result = self.nodes[node](result)
            if node in self.cond:
                node = self.cond[node].get(result, END)
            else:
                node = self.edges.get(node, [END])[0]
        return result

END = "END"
langgraph_mod.StateGraph = _Graph
langgraph_mod.END = END
sys.modules["langgraph.graph"] = langgraph_mod

agents_pkg = ModuleType("legal_ai_system.agents")
agent_nodes_pkg = ModuleType("legal_ai_system.agents.agent_nodes")

class AnalysisNode:
    def __init__(self, topic):
        self.topic = topic

    def __call__(self, text):
        return text


class SummaryNode:
    def __call__(self, text):
        return text

agent_nodes_pkg.AnalysisNode = AnalysisNode
agent_nodes_pkg.SummaryNode = SummaryNode
sys.modules["legal_ai_system.agents"] = agents_pkg
sys.modules["legal_ai_system.agents.agent_nodes"] = agent_nodes_pkg

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Create minimal package hierarchy for relative imports
sys.modules.setdefault("legal_ai_system", ModuleType("legal_ai_system"))
sys.modules.setdefault("legal_ai_system.workflows", ModuleType("legal_ai_system.workflows"))
sys.modules.setdefault("legal_ai_system.workflows.routing", ModuleType("legal_ai_system.workflows.routing"))

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module

doc_mod = _load_module(
    "legal_ai_system.workflows.routing.document_classification_node",
    ROOT / "workflows" / "routing" / "document_classification_node.py",
)
_load_module(
    "legal_ai_system.workflows.routing.analysis_paths",
    ROOT / "workflows" / "routing" / "analysis_paths.py",
)
_load_module(
    "legal_ai_system.workflows.langgraph_setup",
    ROOT / "workflows" / "langgraph_setup.py",
)
aw = _load_module(
    "legal_ai_system.workflows.routing.advanced_workflow",
    ROOT / "workflows" / "routing" / "advanced_workflow.py",
)

DocumentClassificationNode = doc_mod.DocumentClassificationNode


class DummyGraph:
    """Minimal graph implementation for testing routing logic."""

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.cond_edges = {}
        self.entry = None

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, src, dst):
        self.edges.setdefault(src, []).append(dst)

    def add_conditional_edges(self, src, mapping):
        self.cond_edges[src] = mapping

    def set_entry_point(self, name):
        self.entry = name

    def run(self, data):
        node = self.entry
        result = data
        while node != aw.END:
            func = self.nodes[node]
            result = func(result)
            if node in self.cond_edges:
                node = self.cond_edges[node].get(result, aw.END)
            else:
                node = self.edges.get(node, [aw.END])[0]
        return result


@pytest.mark.parametrize(
    "classification,expected",
    [
        ("contract", "contract_result"),
        ("litigation", "litigation_result"),
        ("regulatory", "regulatory_result"),
        ("evidence", "evidence_result"),
    ],
)
def test_routing_paths(monkeypatch, classification, expected):
    monkeypatch.setattr(aw, "StateGraph", DummyGraph)
    monkeypatch.setattr(DocumentClassificationNode, "__call__", lambda self, d: classification)

    graph = aw.build_advanced_legal_workflow()
    result = graph.run("doc")

    assert result == expected
