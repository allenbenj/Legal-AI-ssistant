import shutil
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import AsyncMock

import pytest

# Stub heavy optional dependency before importing service module
import sys

class _DummyFaissIndex:
    def __init__(self, *args, **kwargs) -> None:
        pass

faiss_stub = ModuleType("faiss")
faiss_stub.IndexFlatL2 = _DummyFaissIndex
faiss_stub.IndexIVFFlat = _DummyFaissIndex
faiss_stub.IndexHNSWFlat = _DummyFaissIndex
faiss_stub.IndexPQ = _DummyFaissIndex
faiss_stub.IndexIVFPQ = _DummyFaissIndex
faiss_stub.get_num_gpus = lambda: 0
faiss_stub.StandardGpuResources = object
faiss_stub.index_cpu_to_gpu = lambda *a, **k: _DummyFaissIndex()
sys.modules.setdefault("faiss", faiss_stub)
for name in [
    "asyncpg",
    "aioredis",
    "aioredis.client",
    "aioredis.connection",
    "aioredis.exceptions",
]:
    mod = ModuleType(name)
    if name == "asyncpg":
        mod.Connection = object
    if name == "aioredis":
        mod.Redis = object
    sys.modules.setdefault(name, mod)

sec_mod = ModuleType("legal_ai_system.services.security_manager")
class _DummyUser:
    def __init__(self, user_id="u1"):
        self.user_id = user_id

sec_mod.User = _DummyUser
sec_mod.SecurityManager = object
sys.modules.setdefault("legal_ai_system.services.security_manager", sec_mod)

# Minimal stub for pydantic to avoid import-time errors
pydantic_stub = ModuleType("pydantic")
pydantic_stub.BaseModel = object
pydantic_stub.BaseSettings = object
def _field(default=None, **kwargs):
    return default
pydantic_stub.Field = _field
sys.modules.setdefault("pydantic", pydantic_stub)

for name in [
    "spacy",
    "spacy.language",
    "spacy.tokens",
    "numpy",
    "sklearn",
    "sklearn.metrics",
    "sklearn.metrics.pairwise",
]:
    sys.modules.setdefault(name, ModuleType(name))
sys.modules["sklearn.metrics.pairwise"].cosine_similarity = lambda *a, **k: []
if not hasattr(sys.modules["numpy"], "array"):
    sys.modules["numpy"].array = lambda *a, **k: a
    sys.modules["numpy"].ndarray = object
    sys.modules["numpy"].float32 = float
if not hasattr(sys.modules["spacy.tokens"], "Doc"):
    sys.modules["spacy.tokens"].Doc = object
    sys.modules["spacy.tokens"].Span = object
    sys.modules["spacy.tokens"].SpanGroup = object
if not hasattr(sys.modules["spacy.language"], "Language"):
    sys.modules["spacy.language"].Language = object

from legal_ai_system.services.integration_service import (
    LegalAIIntegrationService,
    ServiceLayerError,
)


class DummyContainer:
    def __init__(self, services=None):
        self.services = services or {}
        self.tasks = []

    async def get_service(self, name):
        return self.services.get(name)

    def add_background_task(self, coro):
        self.tasks.append(coro)


class DummyUser:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id


@pytest.mark.asyncio
async def test_save_uploaded_file_creates_file():
    container = DummyContainer()
    service = LegalAIIntegrationService(container)

    upload_dir = Path("./storage/documents/uploads_service")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)

    file_path, unique = await service._save_uploaded_file(b"data", "test?.txt")
    assert file_path.exists()
    assert file_path.read_bytes() == b"data"
    assert unique.endswith("_test_.txt")

    shutil.rmtree(upload_dir)


@pytest.mark.asyncio
async def test_save_uploaded_file_error(mocker):
    container = DummyContainer()
    service = LegalAIIntegrationService(container)

    mocker.patch("builtins.open", side_effect=IOError("fail"))
    with pytest.raises(ServiceLayerError):
        await service._save_uploaded_file(b"x", "a.txt")


@pytest.mark.asyncio
async def test_create_document_metadata(tmp_path):
    mem = SimpleNamespace(create_session=AsyncMock())
    container = DummyContainer({"memory_manager": mem})
    service = LegalAIIntegrationService(container)
    service.create_document_record = AsyncMock()
    user = DummyUser("u1")

    file_path = tmp_path / "f.txt"
    file_path.write_text("d")

    doc_id, meta = await service._create_document_metadata(file_path, "f.txt", user, {})
    service.create_document_record.assert_awaited_once()
    mem.create_session.assert_awaited_once()
    assert meta["document_id"] == doc_id


@pytest.mark.asyncio
async def test_create_document_metadata_error(tmp_path):
    container = DummyContainer()
    service = LegalAIIntegrationService(container)
    service.create_document_record = AsyncMock(side_effect=Exception("db"))
    user = DummyUser("u")
    file_path = tmp_path / "f.txt"
    file_path.write_text("d")

    with pytest.raises(ServiceLayerError):
        await service._create_document_metadata(file_path, "f.txt", user, {})


@pytest.mark.asyncio
async def test_launch_workflow_adds_task(tmp_path):
    orchestrator = SimpleNamespace(execute_workflow_instance=AsyncMock())
    container = DummyContainer({"workflow_orchestrator": orchestrator})
    service = LegalAIIntegrationService(container)

    await service._launch_workflow(tmp_path / "file.txt", {"document_id": "d"})
    orchestrator.execute_workflow_instance.assert_awaited_once()
    assert container.tasks


@pytest.mark.asyncio
async def test_launch_workflow_error(tmp_path):
    orchestrator = SimpleNamespace(
        execute_workflow_instance=AsyncMock(side_effect=Exception("boom"))
    )
    container = DummyContainer({"workflow_orchestrator": orchestrator})
    service = LegalAIIntegrationService(container)

    with pytest.raises(ServiceLayerError):
        await service._launch_workflow(tmp_path / "file.txt", {"document_id": "d"})

