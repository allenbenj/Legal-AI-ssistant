import pytest

import importlib
import sys
from types import ModuleType

# Provide stub for realtime_analysis_workflow to avoid heavy imports
stub = sys.modules.setdefault(
    "legal_ai_system.services.realtime_analysis_workflow",
    ModuleType("legal_ai_system.services.realtime_analysis_workflow"),
)
if not hasattr(stub, "RealTimeAnalysisWorkflow"):
    stub.RealTimeAnalysisWorkflow = object

module = importlib.import_module("legal_ai_system.services.service_container")
ServiceContainer = importlib.reload(module).ServiceContainer

class DummyService:
    def __init__(self, name: str, calls: list[str]):
        self.name = name
        self.calls = calls

    async def initialize_service(self):
        self.calls.append(self.name)

@pytest.mark.asyncio
async def test_initialization_order_preserved() -> None:
    calls: list[str] = []
    container = ServiceContainer()
    await container.register_service("svc1", factory=lambda sc: DummyService("svc1", calls))
    await container.register_service("svc2", factory=lambda sc: DummyService("svc2", calls))
    await container.register_service("svc3", factory=lambda sc: DummyService("svc3", calls))

    await container.initialize_all_services()

    assert calls == ["svc1", "svc2", "svc3"]
