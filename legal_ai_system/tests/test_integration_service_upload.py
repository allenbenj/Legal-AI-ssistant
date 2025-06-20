import sys
import asyncio
import builtins
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# Stub heavy optional dependencies before importing the service module
for name in [
    "aioredis",
    "aioredis.client",
    "aioredis.connection",
    "aioredis.exceptions",
    "spacy",
    "spacy.language",
    "spacy.tokens",
    "numpy",
    "sklearn",
    "sklearn.metrics",
    "sklearn.metrics.pairwise",
    "faiss",
    "asyncpg",
    "legal_ai_system.utils.user_repository",
    "legal_ai_system.services.security_manager",
    "legal_ai_system.services.workflow_orchestrator",
    "legal_ai_system.services.realtime_analysis_workflow",
    "legal_ai_system.integration_ready.vector_store_enhanced",
    "yaml",
]:
    if name not in sys.modules:
        sys.modules[name] = ModuleType(name)



sys.modules["faiss"].StandardGpuResources = object
sys.modules["faiss"].index_cpu_to_gpu = lambda *a, **k: None
sys.modules["faiss"].IndexFlatL2 = object
sys.modules["faiss"].IndexIDMap = object
sys.modules["faiss"].IndexIVFFlat = object
sys.modules["faiss"].IndexHNSWFlat = object
sys.modules["faiss"].IndexPQ = object
sys.modules["faiss"].IndexIVFPQ = object
sys.modules["faiss"].get_num_gpus = lambda: 0
sys.modules["faiss"].read_index = lambda *a, **k: object()

# Provide placeholder MemoryStore used by memory_manager
sys.modules[
    "legal_ai_system.integration_ready.vector_store_enhanced"
].MemoryStore = object

sys.modules["numpy"].array = lambda *a, **k: a
sys.modules["numpy"].ndarray = object

sys.modules["asyncpg"].Connection = object

sys.modules["aioredis"].Redis = object
sys.modules["aioredis"].StrictRedis = object
exc = ModuleType("exceptions")
exc.RedisError = type("RedisError", (Exception,), {})
exc.TimeoutError = type("TimeoutError", (Exception,), {})
sys.modules["aioredis.exceptions"] = exc
sys.modules["sklearn.metrics.pairwise"].cosine_similarity = lambda *a, **k: []
if not hasattr(sys.modules["spacy.tokens"], "Doc"):
    sys.modules["spacy.tokens"].Doc = object
    sys.modules["spacy.tokens"].Span = object
    sys.modules["spacy.tokens"].SpanGroup = object

sec_mod = sys.modules["legal_ai_system.services.security_manager"]
class User:
    def __init__(self, user_id: str):
        self.user_id = user_id
sec_mod.SecurityManager = object
sec_mod.User = User
sys.modules["legal_ai_system.utils.user_repository"].UserRepository = object


sys.modules["legal_ai_system.services.workflow_orchestrator"].WorkflowOrchestrator = object

from legal_ai_system.services.integration_service import (
    LegalAIIntegrationService,
)
from legal_ai_system.core.unified_exceptions import ServiceLayerError


class DummyContainer:
    def __init__(self, services=None):
        self.services = services or {}
        self.background_tasks = []

    async def get_service(self, name: str):
        return self.services.get(name)

    def add_background_task(self, coro):
        self.background_tasks.append(coro)


class DummyMemoryManager:
    def __init__(self):
        self.sessions = []

    async def create_session(self, session_id, session_name=None, metadata=None):
        self.sessions.append((session_id, session_name, metadata))


class DummyOrchestrator:
    def __init__(self):
        self.calls = []

    def execute_workflow_instance(self, document_path_str: str, custom_metadata: dict):
        self.calls.append((document_path_str, custom_metadata))

        async def _run():
            return None

        return _run()


@pytest.mark.asyncio
async def test_save_uploaded_file_success(tmp_path, monkeypatch):
    container = DummyContainer()
    svc = LegalAIIntegrationService(container)

    # Patch Path used inside helper so files are written to tmp_path
    import legal_ai_system.services.integration_service as module

    orig_path = module.Path

    def patched(path_str: str):
        if path_str == "./storage/documents/uploads_service":
            return tmp_path
        return orig_path(path_str)

    monkeypatch.setattr(module, "Path", patched)

    file_path, unique = await svc._save_uploaded_file(b"data", "test.txt")
    assert file_path.exists()
    assert file_path.read_bytes() == b"data"
    assert unique in file_path.name


@pytest.mark.asyncio
async def test_save_uploaded_file_error(monkeypatch, tmp_path):
    container = DummyContainer()
    svc = LegalAIIntegrationService(container)
    import legal_ai_system.services.integration_service as module

    orig_path = module.Path

    def patched(path_str: str):
        if path_str == "./storage/documents/uploads_service":
            return tmp_path
        return orig_path(path_str)

    monkeypatch.setattr(module, "Path", patched)
    orig_open = builtins.open

    def failing_open(path, mode="r", *a, **k):
        if str(path).startswith(str(tmp_path)):
            raise IOError("fail")
        return orig_open(path, mode, *a, **k)

    monkeypatch.setattr("builtins.open", failing_open)
    with pytest.raises(ServiceLayerError):
        await svc._save_uploaded_file(b"data", "bad.txt")


@pytest.mark.asyncio
async def test_create_document_metadata_success(tmp_path, monkeypatch):
    mem = DummyMemoryManager()
    container = DummyContainer({"memory_manager": mem})
    svc = LegalAIIntegrationService(container)

    async def dummy_create(*a, **k):
        return None

    monkeypatch.setattr(svc, "create_document_record", dummy_create)

    doc_id, meta = await svc._create_document_metadata(
        tmp_path / "file.txt",
        "file.txt",
        SimpleNamespace(user_id="u1"),
        {"a": 1},
    )

    assert doc_id.startswith("doc_serv_")
    assert meta["original_filename"] == "file.txt"
    assert mem.sessions and mem.sessions[0][0] == doc_id


@pytest.mark.asyncio
async def test_create_document_metadata_error(monkeypatch, tmp_path):
    container = DummyContainer({"memory_manager": DummyMemoryManager()})
    svc = LegalAIIntegrationService(container)

    async def failing(*a, **k):
        raise RuntimeError("db fail")

    monkeypatch.setattr(svc, "create_document_record", failing)

    with pytest.raises(ServiceLayerError):
        await svc._create_document_metadata(
            "file.txt",
            tmp_path / "file.txt",
            SimpleNamespace(user_id="u1"),
            {},
        )


@pytest.mark.asyncio
async def test_launch_workflow_success(tmp_path):
    orch = DummyOrchestrator()
    container = DummyContainer({"workflow_orchestrator": orch})
    svc = LegalAIIntegrationService(container)

    await svc._launch_workflow(
        tmp_path / "doc.txt",
        {"document_id": "d"},
        progress_cb=None,
    )

    assert len(container.background_tasks) == 1
    coro = container.background_tasks[0]
    assert asyncio.iscoroutine(coro)


@pytest.mark.asyncio
async def test_launch_workflow_no_service(tmp_path):
    container = DummyContainer({})
    svc = LegalAIIntegrationService(container)

    with pytest.raises(ServiceLayerError):
        await svc._launch_workflow(
            tmp_path / "doc.txt",
            {"document_id": "d"},
            progress_cb=None,
        )


@pytest.mark.asyncio
async def test_launch_workflow_error(monkeypatch, tmp_path):
    orch = DummyOrchestrator()

    def failing(*a, **k):
        raise RuntimeError("boom")

    orch.execute_workflow_instance = failing  # type: ignore
    container = DummyContainer({"workflow_orchestrator": orch})
    svc = LegalAIIntegrationService(container)

    with pytest.raises(ServiceLayerError):
        await svc._launch_workflow(
            tmp_path / "doc.txt",
            {"document_id": "d"},
            progress_cb=None,
        )


@pytest.mark.asyncio
async def test_upload_and_process_document_success(tmp_path):
    orch = DummyOrchestrator()
    container = DummyContainer({"workflow_orchestrator": orch})
    svc = LegalAIIntegrationService(container)

    svc._save_uploaded_file = AsyncMock(return_value=(tmp_path / "f.txt", "f.txt"))
    svc._create_document_metadata = AsyncMock(return_value=("docid", {"document_id": "docid"}))
    svc._launch_workflow = AsyncMock()

    user = sec_mod.User("u1")
    result = await svc.upload_and_process_document(b"x", "f.txt", user, progress_cb=None)

    assert result["document_id"] == "docid"
    svc._save_uploaded_file.assert_awaited_once()
    svc._create_document_metadata.assert_awaited_once()
    svc._launch_workflow.assert_awaited_once()


@pytest.mark.asyncio
async def test_upload_and_process_document_error(tmp_path):
    container = DummyContainer()
    svc = LegalAIIntegrationService(container)

    svc._save_uploaded_file = AsyncMock(side_effect=ServiceLayerError("fail"))

    user = sec_mod.User("u1")
    with pytest.raises(ServiceLayerError):
        await svc.upload_and_process_document(b"x", "f.txt", user, progress_cb=None)

