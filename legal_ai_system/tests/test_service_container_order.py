import asyncio
import pytest

from legal_ai_system.services.service_container import ServiceContainer

class DummyA:
    def __init__(self):
        self.initialized = False

    async def initialize_service(self):
        await asyncio.sleep(0)
        self.initialized = True

class DummyB:
    def __init__(self):
        self.initialized = False

    async def initialize_service(self):
        await asyncio.sleep(0)
        self.initialized = True

@pytest.mark.asyncio
async def test_initialization_order():
    container = ServiceContainer()

    await container.register_service("a", factory=lambda sc: DummyA())
    await container.register_service("b", factory=lambda sc: DummyB())

    await container.initialize_all_services()

    assert container._initialization_order == ["a", "b"]
    assert container._service_states["a"].name == "INITIALIZED"
    assert container._service_states["b"].name == "INITIALIZED"
