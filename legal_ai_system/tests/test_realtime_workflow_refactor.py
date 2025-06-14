import asyncio
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

import pytest

# Stub heavy optional dependencies before importing workflow module
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
    "sklearn.feature_extraction.text",
    "sklearn.linear_model",
    "pydantic",
    "fastapi",
    "prometheus_client",
    "uvicorn",
]:
    if name not in sys.modules:
        sys.modules[name] = ModuleType(name)
    if name == "pydantic":
        sys.modules[name].BaseModel = object
        sys.modules[name].BaseSettings = object
        sys.modules[name].Field = lambda *a, **k: k.get("default") if "default" in k else (a[0] if a else None)
    if name == "fastapi":
        sys.modules[name].FastAPI = object
    if name == "prometheus_client":
        sys.modules[name].Counter = lambda *a, **k: object()
        sys.modules[name].Histogram = lambda *a, **k: object()
        sys.modules[name].Gauge = lambda *a, **k: object()
        sys.modules[name].make_asgi_app = lambda *a, **k: None
    if name == "uvicorn":
        sys.modules[name].run = lambda *a, **k: None

class _RedisError(Exception):
    pass

sys.modules["aioredis"].Redis = object
sys.modules["aioredis"].StrictRedis = object
exc = ModuleType("exceptions")
exc.RedisError = _RedisError
exc.TimeoutError = TimeoutError = type("TimeoutError", (Exception,), {})
sys.modules["aioredis.exceptions"] = exc
sys.modules["sklearn.metrics.pairwise"].cosine_similarity = lambda *a, **k: []
sys.modules.setdefault("sklearn.feature_extraction", ModuleType("sklearn.feature_extraction"))
sys.modules["sklearn.feature_extraction"].text = ModuleType("text")
sys.modules["sklearn.feature_extraction"].text.TfidfVectorizer = object
sys.modules["sklearn.feature_extraction.text"] = sys.modules["sklearn.feature_extraction"].text
sys.modules.setdefault("sklearn.linear_model", ModuleType("sklearn.linear_model"))
sys.modules["sklearn.linear_model"].LogisticRegression = object
if not hasattr(sys.modules["numpy"], "array"):
    sys.modules["numpy"].array = lambda *a, **k: a
if not hasattr(sys.modules["numpy"], "ndarray"):
    class _FakeNdarray(list):
        ndim = 1

    sys.modules["numpy"].ndarray = _FakeNdarray
if not hasattr(sys.modules["spacy.tokens"], "Doc"):
    sys.modules["spacy.tokens"].Doc = object
    sys.modules["spacy.tokens"].Span = object
    sys.modules["spacy.tokens"].SpanGroup = object
if not hasattr(sys.modules["spacy.language"], "Language"):
    sys.modules["spacy.language"].Language = object

try:
    from legal_ai_system.services.realtime_analysis_workflow import (
        RealTimeAnalysisWorkflow,
        RealTimeAnalysisResult,
    )
except Exception:  # pragma: no cover - optional dependency may be missing
    sys.modules.pop("legal_ai_system.services.service_container", None)
    pytest.skip("workflow module unavailable", allow_module_level=True)


class DummyWorkflow(RealTimeAnalysisWorkflow):
    def __init__(self, queue=None):
        # Set minimal attributes without calling super().__init__
        self.processing_lock = asyncio.Semaphore(1)
        self.logger = MagicMock()
        self.max_concurrent_documents = 1
        self.auto_optimization_threshold = 1000
        self.documents_processed = 0
        self.progress_callbacks = []
        self.update_callbacks = []
        self.enable_real_time_sync = False
        self.task_queue = queue

        self.document_processor = SimpleNamespace(
            process=AsyncMock(return_value=SimpleNamespace(data=SimpleNamespace(content="txt", success=True)))
        )
        self.document_rewriter = SimpleNamespace(
            rewrite_text=AsyncMock(return_value=SimpleNamespace(corrected_text="txt"))
        )
        self.hybrid_extractor = SimpleNamespace(
            extract_from_document=AsyncMock(return_value=SimpleNamespace(validated_entities=[], targeted_extractions={}, document_id="d")),
            initialize=AsyncMock(),
            close=AsyncMock(),
        )
        self.ontology_extractor = SimpleNamespace(
            process=AsyncMock(return_value=SimpleNamespace(entities=[], relationships=[]))
        )
        self.graph_manager = SimpleNamespace(
            initialize_service=AsyncMock(),
            close=AsyncMock(),
            get_realtime_stats=AsyncMock(return_value={}),
        )
        self.vector_store = SimpleNamespace(
            initialize=AsyncMock(),
            close=AsyncMock(),
            add_vector_async=AsyncMock(return_value=None),
            optimize_performance=AsyncMock(return_value={}),
            get_service_status=AsyncMock(return_value={}),
        )
        self.reviewable_memory = SimpleNamespace(
            initialize=AsyncMock(),
            close=AsyncMock(),
            process_extraction_result=AsyncMock(return_value={}),
            get_pending_reviews_async=AsyncMock(return_value=[]),
            get_review_stats_async=AsyncMock(return_value={}),
            submit_review_decision_async=AsyncMock(return_value=True),
        )
        self.policy_learner = SimpleNamespace(
            should_run_step=lambda *a, **k: True,
            update_agent_stats=lambda *a, **k: None,
            record_step=lambda *a, **k: None,
            predict_concurrency=lambda *a, **k: 1,
        )

        # Patch internal helpers
        self._notify_progress = AsyncMock()
        self._notify_update = AsyncMock()
        self._update_knowledge_graph_realtime = AsyncMock(return_value={})
        self._update_vector_store_realtime = AsyncMock(return_value={})
        self._integrate_with_memory = AsyncMock(return_value={})
        self._validate_extraction_quality = AsyncMock(return_value={})
        self._calculate_confidence_scores = MagicMock(return_value={})
        self._get_sync_status = AsyncMock(return_value={})
        self._update_performance_stats = AsyncMock()
        self._auto_optimize_system = AsyncMock()
        self._create_legal_document = MagicMock(
            side_effect=lambda result, path, doc_id, text_override=None: SimpleNamespace(
                id=doc_id,
                file_path=path,
                content=text_override or result.get("content", ""),
                metadata={"processing_result": result},
            )
        )
        self._extract_text_from_result = MagicMock(return_value="txt")


@pytest.mark.asyncio
async def test_process_document_realtime_returns_result_structure() -> None:
    wf = DummyWorkflow()
    result = await wf.process_document_realtime("sample.txt")

    assert isinstance(result, RealTimeAnalysisResult)
    data = result.to_dict()
    expected_keys = {
        "document_path",
        "document_id",
        "processing_times",
        "total_processing_time",
        "confidence_scores",
        "validation_results",
        "sync_status",
        "graph_updates",
        "vector_updates",
        "memory_updates",
        "text_rewriting",
    }
    assert expected_keys.issubset(set(data.keys()))
    assert result.document_id


@pytest.mark.asyncio
async def test_process_document_realtime_uses_provided_document_id() -> None:
    wf = DummyWorkflow()
    result = await wf.process_document_realtime(
        "sample.txt", document_id="custom_id"
    )

    assert result.document_id == "custom_id"


@pytest.mark.asyncio
async def test_process_document_realtime_updates_graph_and_vectors() -> None:
    wf = DummyWorkflow()
    await wf.process_document_realtime("sample.txt")

    wf._update_knowledge_graph_realtime.assert_awaited()
    wf._update_vector_store_realtime.assert_awaited()


@pytest.mark.asyncio
async def test_process_document_realtime_invokes_rewriter() -> None:
    wf = DummyWorkflow()
    await wf.process_document_realtime("sample.txt")

    wf.document_rewriter.rewrite_text.assert_awaited()


@pytest.mark.asyncio
async def test_process_document_realtime_queues_job_when_queue_available() -> None:
    queue = MagicMock()
    queue.enqueue = MagicMock(return_value="job")
    wf = DummyWorkflow(queue=queue)

    result = await wf.process_document_realtime("sample.txt")

    assert result == "job"
    queue.enqueue.assert_called_once()
    wf._notify_progress.assert_awaited_with("queued", 0.0)


@pytest.mark.asyncio
async def test_process_document_realtime_emits_completion_update() -> None:
    wf = DummyWorkflow()
    await wf.process_document_realtime("sample.txt")

    wf._notify_update.assert_any_await(
        "workflow_completed", pytest.ANY
    )


@pytest.mark.asyncio
async def test_process_document_realtime_emits_failure_update() -> None:
    wf = DummyWorkflow()
    wf._run_realtime_pipeline = AsyncMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        await wf.process_document_realtime("sample.txt")

    wf._notify_update.assert_any_await(
        "workflow_failed", pytest.ANY
    )



