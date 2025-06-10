from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import sys

sys.modules.setdefault("fitz", ModuleType("fitz"))
sys.modules.setdefault("pytesseract", ModuleType("pytesseract"))
pil = ModuleType("PIL")
sys.modules.setdefault("PIL", pil)
sys.modules.setdefault("PIL.Image", ModuleType("PIL.Image"))
rts = ModuleType("legal_ai_system.agents.agent_nodes")
rts.AnalysisNode = lambda topic: (lambda x: f"analysis:{topic}")
rts.SummaryNode = lambda: (lambda x: "summary")
sys.modules.setdefault("legal_ai_system.agents.agent_nodes", rts)
sys.modules.setdefault(
    "legal_ai_system.core.unified_services", ModuleType("unified_services")
)
sys.modules.setdefault(
    "legal_ai_system.integration_ready.vector_store_enhanced", ModuleType("vec")
)
rtwf_mod = ModuleType("legal_ai_system.services.realtime_analysis_workflow")
class DummyRTWorkflow:
    def __init__(self, *a, **k):
        self.initialize = AsyncMock()
        self.process_document_realtime = AsyncMock(
            return_value=SimpleNamespace(document_id="dummy")
        )
        self.register_progress_callback = MagicMock()

rtwf_mod.RealTimeAnalysisWorkflow = DummyRTWorkflow
rtwf_mod.RealTimeAnalysisResult = SimpleNamespace
sys.modules.setdefault(
    "legal_ai_system.services.realtime_analysis_workflow", rtwf_mod
)

import importlib
import pytest


@pytest.mark.asyncio
async def test_initialize_builds_graph_and_initializes_services() -> None:
    service_container = SimpleNamespace(
        initialize_all_services=AsyncMock(),
        get_service=AsyncMock(return_value=None),
    )

    workflow = SimpleNamespace(initialize=AsyncMock(), register_progress_callback=MagicMock())
    module = importlib.reload(
        importlib.import_module("legal_ai_system.services.workflow_orchestrator")
    )
    WorkflowOrchestrator = module.WorkflowOrchestrator
    orchestrator = WorkflowOrchestrator(service_container)
    orchestrator.workflow = workflow
    graph_obj = SimpleNamespace()
    orchestrator.graph_builder = MagicMock(return_value=graph_obj)

    await orchestrator.initialize_service()

    service_container.initialize_all_services.assert_awaited()
    workflow.initialize.assert_awaited()
    orchestrator.graph_builder.assert_called_once_with("default")
    assert orchestrator._graph is graph_obj


@pytest.mark.asyncio
async def test_execute_advanced_workflow_runs_graph(tmp_path) -> None:
    service_container = SimpleNamespace(
        initialize_all_services=AsyncMock(),
        get_service=AsyncMock(return_value=None),
    )

    workflow = SimpleNamespace(
        initialize=AsyncMock(),
        process_document_realtime=AsyncMock(
            return_value=SimpleNamespace(document_id="doc123")
        ),
        register_progress_callback=MagicMock(),
    )
    module = importlib.reload(
        importlib.import_module("legal_ai_system.services.workflow_orchestrator")
    )
    WorkflowOrchestrator = module.WorkflowOrchestrator
    orchestrator = WorkflowOrchestrator(service_container)
    orchestrator.workflow = workflow
    graph_obj = SimpleNamespace(run=MagicMock())
    orchestrator.graph_builder = MagicMock(return_value=graph_obj)

    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello")

    with patch(
        "legal_ai_system.services.workflow_orchestrator.extract_text",
        return_value="content",
    ):
        await orchestrator.initialize_service()
        result_id = await orchestrator.execute_workflow_instance(str(file_path))

    workflow.process_document_realtime.assert_awaited_with(
        document_path=str(file_path),
    )
    graph_obj.run.assert_called_once_with("content")
    assert result_id == "doc123"

