import sys
import asyncio
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pyd = ModuleType("pydantic")
pyd.BaseModel = object
pyd.BaseSettings = object
def _field(default=None, **kwargs):
    return default
pyd.Field = _field
sys.modules["pydantic"] = pyd
sys.modules.setdefault(
    "legal_ai_system.analytics.keyword_extractor", ModuleType("keyword_extractor")
).extract_keywords = lambda text: []
if "numpy" not in sys.modules:
    numpy_stub = ModuleType("numpy")
    numpy_stub.array = lambda *a, **k: a
    numpy_stub.ndarray = list
    sys.modules["numpy"] = numpy_stub
if "sklearn" not in sys.modules:
    skl = ModuleType("sklearn")
    metrics = ModuleType("metrics")
    metrics.pairwise = ModuleType("pairwise")
    metrics.pairwise.cosine_similarity = lambda *a, **k: []
    skl.metrics = metrics
    fe = ModuleType("feature_extraction")
    text_mod = ModuleType("text")
    text_mod.TfidfVectorizer = object
    fe.text = text_mod
    skl.feature_extraction = fe
    linear_model = ModuleType("linear_model")
    linear_model.LogisticRegression = object
    skl.linear_model = linear_model
    sys.modules["sklearn"] = skl
    sys.modules["sklearn.metrics"] = metrics
    sys.modules["sklearn.metrics.pairwise"] = metrics.pairwise
    sys.modules["sklearn.feature_extraction"] = fe
    sys.modules["sklearn.feature_extraction.text"] = text_mod
    sys.modules["sklearn.linear_model"] = linear_model
prom = ModuleType("prometheus_client")
prom.Counter = prom.Histogram = prom.Gauge = object
prom.start_http_server = lambda *a, **k: None
sys.modules["prometheus_client"] = prom

from legal_ai_system.services.realtime_analysis_workflow import RealTimeAnalysisWorkflow

class DummyWorkflow(RealTimeAnalysisWorkflow):
    def __init__(self):
        # minimal setup without calling super
        self.graph_manager = SimpleNamespace(
            add_entity_realtime=AsyncMock(return_value="e"),
            add_relationship_realtime=AsyncMock(return_value="r"),
        )
        self._convert_to_extracted_entity = lambda e: e
        self.min_entity_confidence_for_kg = 0.6
        self.logger = SimpleNamespace(error=lambda *a, **k: None)

@pytest.mark.asyncio
async def test_update_knowledge_graph_filters_and_deduplicates():
    wf = DummyWorkflow()
    hybrid = SimpleNamespace(
        validated_entities=[
            SimpleNamespace(entity_text="A", consensus_type="CASE", confidence=0.7),
            SimpleNamespace(entity_text="A", consensus_type="CASE", confidence=0.9),
            SimpleNamespace(entity_text="low", consensus_type="CASE", confidence=0.4),
        ]
    )
    ontology = SimpleNamespace(
        entities=[
            SimpleNamespace(entity_text="B", entity_type="ORGANIZATION", confidence=0.65),
            SimpleNamespace(entity_text="A", entity_type="CASE", confidence=0.8),
        ],
        relationships=[
            SimpleNamespace(source_entity="1", target_entity="2", relationship_type="FILED_BY", confidence=0.7),
            SimpleNamespace(source_entity="1", target_entity="2", relationship_type="FILED_BY", confidence=0.7),
            SimpleNamespace(source_entity="1", target_entity="2", relationship_type="FILED_BY", confidence=0.4),
        ],
    )

    summary = await wf._update_knowledge_graph_realtime(hybrid, ontology, "doc")

    assert summary == {"nodes_added": 2, "relationships_added": 1}
    assert wf.graph_manager.add_entity_realtime.await_count == 2
    assert wf.graph_manager.add_relationship_realtime.await_count == 1
