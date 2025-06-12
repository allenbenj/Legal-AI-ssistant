import sys
import types
import pytest

# Stub heavy dependencies before importing the vector store
sys.modules.setdefault(
    "legal_ai_system.core.enhanced_persistence", types.ModuleType("dummy_ep")
)
sys.modules.setdefault(
    "legal_ai_system.services.service_container", types.ModuleType("dummy_sc")
)
sys.modules.setdefault(
    "legal_ai_system.services.security_manager", types.ModuleType("dummy_sec")
)
sys.modules.setdefault(
    "legal_ai_system.utils.user_repository", types.ModuleType("dummy_user_repo")
)

from legal_ai_system.core.vector_store import VectorStore, VectorMetadata, IndexType


from legal_ai_system.core.vector_store import EmbeddingProviderVS


class DummyProvider(EmbeddingProviderVS):
    def __init__(self):
        super().__init__(model_name="dummy")
        self.dimension = 8

    async def initialize(self):
        pass

    async def embed_texts(self, texts, batch_size=None):
        return [[0.0] * self.dimension for _ in texts]


@pytest.mark.asyncio
async def test_get_metadata_by_faiss_internal_id(tmp_path):
    provider = DummyProvider()
    store = VectorStore(str(tmp_path), provider, default_index_type=IndexType.FLAT)
    await store.initialize()

    meta = VectorMetadata(
        faiss_id=1,
        vector_id="vec1",
        document_id="doc",
        content_hash="h",
        content_preview="",
        vector_norm=0.0,
        dimension=provider.dimension,
    )
    store.metadata_mem_cache["vec1"] = meta
    store.faissid_to_vectorid_doc[1] = "vec1"
    store.vectorid_to_faissid_doc["vec1"] = 1

    async def fake_get_vector_id(fid, index_target):
        return store.faissid_to_vectorid_doc.get(fid)

    store._get_vector_id_by_faiss_id_async = fake_get_vector_id

    result = await store._get_metadata_by_faiss_internal_id_async(1, "document")
    assert result == meta


@pytest.mark.asyncio
async def test_get_metadata_by_faiss_internal_id_missing(tmp_path):
    provider = DummyProvider()
    store = VectorStore(str(tmp_path), provider, default_index_type=IndexType.FLAT)
    await store.initialize()

    async def fake_get_vector_id(fid, index_target):
        return None

    store._get_vector_id_by_faiss_id_async = fake_get_vector_id

    result = await store._get_metadata_by_faiss_internal_id_async(2, "document")
    assert result is None

