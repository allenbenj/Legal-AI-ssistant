import asyncio
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import List, Any

import pytest
import sys

for name in [
    "aioredis",
    "aioredis.client",
    "aioredis.connection",
    "aioredis.exceptions",
    "fitz",
    "pytesseract",
    "PIL",
    "PIL.Image",
    "spacy",
    "spacy.language",
    "spacy.tokens",
    "numpy",
    "sklearn",
    "sklearn.metrics",
    "sklearn.metrics.pairwise",
    "aiofiles",
    "aiofiles.os",
    "prometheus_client",
]:
    if name not in sys.modules:
        sys.modules[name] = ModuleType(name)

if "prometheus_client" in sys.modules:
    pc = sys.modules["prometheus_client"]
    pc.Counter = lambda *a, **k: None
    pc.Histogram = lambda *a, **k: None
    pc.Gauge = lambda *a, **k: None
    pc.start_http_server = lambda *a, **k: None

class _RedisError(Exception):
    pass

sys.modules["aioredis"].Redis = object
sys.modules["aioredis"].StrictRedis = object
exc = ModuleType("exceptions")
exc.RedisError = _RedisError
exc.TimeoutError = type("TimeoutError", (Exception,), {})
sys.modules["aioredis.exceptions"] = exc
sys.modules["sklearn.metrics.pairwise"].cosine_similarity = lambda *a, **k: []
if not hasattr(sys.modules["numpy"], "array"):
    sys.modules["numpy"].array = lambda *a, **k: a
if not hasattr(sys.modules["spacy.tokens"], "Doc"):
    sys.modules["spacy.tokens"].Doc = object
    sys.modules["spacy.tokens"].Span = object
    sys.modules["spacy.tokens"].SpanGroup = object
if not hasattr(sys.modules["spacy.language"], "Language"):
    sys.modules["spacy.language"].Language = object

class _Graph:
    def __init__(self) -> None:
        self.inputs: List[str] = []

    def add_node(self, *a, **k) -> None:
        pass

    def set_entry_point(self, *a, **k) -> None:
        pass

    def add_edge(self, *a, **k) -> None:
        pass

    def run(self, text: str) -> None:
        self.inputs.append(text)


stub_langgraph = ModuleType("legal_ai_system.workflows.langgraph_setup")
stub_langgraph.StateGraph = _Graph
stub_langgraph.END = "END"
stub_langgraph.build_graph = lambda topic: _Graph()
sys.modules["legal_ai_system.workflows.langgraph_setup"] = stub_langgraph

stub_rt = ModuleType("legal_ai_system.services.realtime_analysis_workflow")

@dataclass
class StubResult:
    document_path: str
    document_id: str
    processing_times: dict = field(default_factory=dict)
    total_processing_time: float = 0.0
    text_rewriting: dict = field(default_factory=dict)
    confidence_scores: dict = field(default_factory=dict)
    validation_results: dict = field(default_factory=dict)
    sync_status: dict = field(default_factory=dict)
    graph_updates: dict = field(default_factory=dict)
    vector_updates: dict = field(default_factory=dict)
    memory_updates: dict = field(default_factory=dict)

    def to_dict(self):
        return self.__dict__

class StubWorkflow:
    async def initialize(self) -> None:
        pass

    async def process_document_realtime(self, document_path: str, **kwargs):
        doc_id = kwargs.get("document_id") or document_path
        return StubResult(document_id=doc_id, document_path=document_path)

stub_rt.RealTimeAnalysisWorkflow = StubWorkflow
stub_rt.RealTimeAnalysisResult = StubResult
sys.modules["legal_ai_system.services.realtime_analysis_workflow"] = stub_rt

if 'pydantic' not in sys.modules:
    pydantic_stub = ModuleType('pydantic')
    pydantic_stub.BaseModel = object
    pydantic_stub.BaseSettings = object
    def _field(default=None, **kwargs):
        return default
    pydantic_stub.Field = _field
    sys.modules['pydantic'] = pydantic_stub

from legal_ai_system.workflows.case_workflow_state import CaseWorkflowState

# Patch heavy dependencies before importing the orchestrator module
orc_mod = pytest.importorskip("legal_ai_system.services.workflow_orchestrator")
OrchestratorClass = orc_mod.WorkflowOrchestrator


class DummyContainer:
    def __init__(self) -> None:
        self._services = {}


class DummyWorkflow:
    """Minimal workflow returning a generated document id."""
    def __init__(self, *a, **k) -> None:
        pass

    async def initialize(self) -> None:
        pass

    async def process_document_realtime(self, document_path: str, **kwargs):
        doc_id = kwargs.get("document_id") or document_path
        return SimpleNamespace(document_id=doc_id)


class DummyGraph:
    def __init__(self) -> None:
        self.inputs: List[Any] = []

    def run(self, state: Any) -> None:
        self.inputs.append(state)


def test_case_workflow_state_basic_operations():
    state = CaseWorkflowState(case_id="c1")
    state.process_new_document("d1", "text1")
    state.update_case_state({"a": 1})
    assert state.get_case_context() == "text1"
    assert state.state_data["a"] == 1


@pytest.mark.asyncio
async def test_workflow_orchestrator_state_persistence(tmp_path, monkeypatch):
    file1 = tmp_path / "a.txt"
    file2 = tmp_path / "b.txt"
    file1.write_text("one")
    file2.write_text("two")

    graph = DummyGraph()

    monkeypatch.setattr(orc_mod, "RealTimeAnalysisWorkflow", DummyWorkflow)
    monkeypatch.setattr(orc_mod, "build_graph", lambda topic: graph)
    monkeypatch.setattr(orc_mod, "extract_text", lambda p: p.read_text())

    orch = OrchestratorClass(service_container=DummyContainer())
    state = CaseWorkflowState(case_id="case")

    await orch.execute_workflow_instance(str(file1), case_state=state)
    await orch.execute_workflow_instance(str(file2), case_state=state)

    assert state.documents and len(state.documents) == 2
    assert state.get_case_context() == "one\ntwo"
    assert graph.inputs == [state, state]
