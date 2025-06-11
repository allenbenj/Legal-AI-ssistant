from types import ModuleType, SimpleNamespace
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
    "pydantic",
    "prometheus_client",
]:
    if name not in sys.modules:
        sys.modules[name] = ModuleType(name)

class _RedisError(Exception):
    pass

sys.modules["aioredis"].Redis = object
sys.modules["aioredis"].StrictRedis = object
exc = ModuleType("exceptions")
exc.RedisError = _RedisError
exc.TimeoutError = TimeoutError = type("TimeoutError", (Exception,), {})
sys.modules["aioredis.exceptions"] = exc
sys.modules["sklearn.metrics.pairwise"].cosine_similarity = lambda *a, **k: []
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

pydantic_stub = sys.modules.setdefault("pydantic", ModuleType("pydantic"))
pydantic_stub.BaseModel = object
pydantic_stub.BaseSettings = object
def _field(default=None, **kwargs):
    return default
pydantic_stub.Field = _field

if "prometheus_client" in sys.modules:
    pc = sys.modules["prometheus_client"]
    pc.Counter = lambda *a, **k: None
    pc.Histogram = lambda *a, **k: None
    pc.Gauge = lambda *a, **k: None
    pc.start_http_server = lambda *a, **k: None

qc_mod = ModuleType("legal_ai_system.analytics.quality_classifier")
class PreprocessingErrorPredictor:
    def predict_risk(self, doc):
        return 0.0
qc_mod.PreprocessingErrorPredictor = PreprocessingErrorPredictor
sys.modules.setdefault("legal_ai_system.analytics.quality_classifier", qc_mod)
analytics_pkg = ModuleType("legal_ai_system.analytics")
analytics_pkg.extract_keywords = lambda *a, **k: []
analytics_pkg.quality_classifier = qc_mod
sys.modules.setdefault("legal_ai_system.analytics", analytics_pkg)

try:
    from legal_ai_system.services.realtime_analysis_workflow import (
        RealTimeAnalysisWorkflow,
    )
except Exception:  # pragma: no cover - optional dependency may be missing
    sys.modules.pop("legal_ai_system.services.service_container", None)
    pytest.skip("workflow module unavailable", allow_module_level=True)

from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_update_vector_store_uses_chunks():
    wf = RealTimeAnalysisWorkflow()
    wf.vector_store = SimpleNamespace(
        add_vector_async=AsyncMock(),
        flush_updates=AsyncMock(),
    )
    hybrid = SimpleNamespace(validated_entities=[], targeted_extractions={}, document_id="d")
    doc_result = SimpleNamespace(text_chunks=["a", "b"], text_content="a b")

    await wf._update_vector_store_realtime(hybrid, doc_result, "doc")

    assert wf.vector_store.add_vector_async.await_count == 2
    wf.vector_store.flush_updates.assert_awaited()


@pytest.mark.asyncio
async def test_update_vector_store_token_fallback():
    wf = RealTimeAnalysisWorkflow()
    wf.vector_store = SimpleNamespace(
        add_vector_async=AsyncMock(),
        flush_updates=AsyncMock(),
    )
    hybrid = SimpleNamespace(validated_entities=[], targeted_extractions={}, document_id="d")
    text = " ".join(f"w{i}" for i in range(1100))
    doc_result = SimpleNamespace(text_content=text)

    await wf._update_vector_store_realtime(hybrid, doc_result, "doc")

    assert wf.vector_store.add_vector_async.await_count == 2
