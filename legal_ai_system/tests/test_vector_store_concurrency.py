import asyncio
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
async def test_concurrent_metadata_updates(tmp_path):
    provider = DummyProvider()
    store = VectorStore(str(tmp_path), provider, default_index_type=IndexType.FLAT)
    await store.initialize()

    meta = VectorMetadata(
        faiss_id=0,
        vector_id="vec1",
        document_id="doc",
        content_hash="h",
        content_preview="",
        vector_norm=0.0,
        dimension=provider.dimension,
    )
    await store._store_metadata_async(meta)

    async def worker(i):
        await store.update_vector_metadata_async("vec1", {"custom_field": i})

    await asyncio.gather(*(worker(i) for i in range(5)))

    final = store.metadata_mem_cache["vec1"]
    assert final.custom_metadata["custom_field"] in range(5)
    assert final.access_count == 5
