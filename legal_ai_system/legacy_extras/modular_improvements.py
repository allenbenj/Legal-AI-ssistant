class ProcessingCache:
    """Minimal async cache stub for tests."""
    def __init__(self) -> None:
        self.storage = {}

    async def get(self, document_id, key):
        return self.storage.get((document_id, key))

    async def set(self, document_id, key, value):
        self.storage[(document_id, key)] = value
