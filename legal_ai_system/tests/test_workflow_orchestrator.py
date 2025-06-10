import asyncio
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

# Stub heavy optional dependencies before importing the module
for name in ["fitz", "pytesseract", "PIL", "PIL.Image"]:
    sys.modules.setdefault(name, ModuleType(name))

fastapi_stub = ModuleType("fastapi")
class _WS:  # simple placeholder
    pass
fastapi_stub.WebSocket = _WS
fastapi_stub.WebSocketDisconnect = Exception
sys.modules.setdefault("fastapi", fastapi_stub)

pydantic_stub = ModuleType("pydantic")
pydantic_stub.BaseModel = object
pydantic_stub.BaseSettings = object
def _field(default=None, **kwargs):
    return default
pydantic_stub.Field = _field
sys.modules.setdefault("pydantic", pydantic_stub)

rt_stub = ModuleType("legal_ai_system.services.realtime_analysis_workflow")
class _StubWF:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def initialize(self) -> None:
        pass

    def register_progress_callback(self, cb):
        pass

    async def process_document_realtime(self, document_path: str, **kwargs):
        return SimpleNamespace(document_id="stub")

rt_stub.RealTimeAnalysisWorkflow = _StubWF
sys.modules.setdefault(
    "legal_ai_system.services.realtime_analysis_workflow", rt_stub
)

int_stub = ModuleType("legal_ai_system.services.integration_service")
int_stub.LegalAIIntegrationService = None
sys.modules.setdefault(
    "legal_ai_system.services.integration_service", int_stub
)

from legal_ai_system.services.workflow_orchestrator import WorkflowOrchestrator


class DummyWorkflow:
    def __init__(self) -> None:
        self.initialized = False
        self.progress_callbacks = []
        self.process_called = False

    async def initialize(self) -> None:
        self.initialized = True

    def register_progress_callback(self, cb):
        self.progress_callbacks.append(cb)

    async def process_document_realtime(self, document_path: str, **kwargs):
        self.process_called = True
        for cb in self.progress_callbacks:
            await cb("running", 0.5)
        return SimpleNamespace(document_id="doc123")


class DummyGraph:
    def __init__(self) -> None:
        self.ran = False

    def run(self, text: str) -> None:
        self.ran = True


def dummy_builder(topic: str) -> DummyGraph:
    return DummyGraph()


class DummyWS:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    async def broadcast(self, topic: str, message: Any) -> None:
        self.events.append((topic, message))


class DummyContainer:
    def __init__(self, services=None) -> None:
        self.services = services or {}
        self.init_called = False

    async def get_service(self, name: str):
        return self.services.get(name)

    async def initialize_all_services(self):
        self.init_called = True


@pytest.mark.asyncio
async def test_orchestrator_initialization_registers_callback():
    wf = DummyWorkflow()
    ws = DummyWS()
    container = DummyContainer({"websocket_manager": ws})
    orch = WorkflowOrchestrator(container)
    orch.workflow = wf
    orch.graph_builder = dummy_builder

    await orch.initialize_service()

    assert wf.initialized
    assert container.init_called
    assert wf.progress_callbacks
    assert orch._graph is not None


@pytest.mark.asyncio
async def test_execute_workflow_instance_runs_graph(tmp_path):
    wf = DummyWorkflow()
    graph = DummyGraph()

    def build(_topic: str) -> DummyGraph:
        return graph

    ws = DummyWS()
    container = DummyContainer({"websocket_manager": ws})
    orch = WorkflowOrchestrator(container)
    orch.workflow = wf
    orch.graph_builder = build

    await orch.initialize_service()

    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello")

    # Patch extract_text to avoid heavy dependencies
    orch_module = __import__("legal_ai_system.services.workflow_orchestrator", fromlist=["extract_text"])
    setattr(orch_module, "extract_text", lambda p: "text")

    doc_id = await orch.execute_workflow_instance(str(file_path))

    assert doc_id == "doc123"
    assert wf.process_called
    assert graph.ran
    assert ws.events

