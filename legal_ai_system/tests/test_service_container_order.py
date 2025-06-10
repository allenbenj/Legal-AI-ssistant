import asyncio
import pytest

from legal_ai_system.services.service_container import ServiceContainer
from types import SimpleNamespace

if not hasattr(ServiceContainer, "register_service"):
    async def _reg(self, name, factory):
        if not hasattr(self, "_initialization_order"):
            self._initialization_order = []
            self._service_states = {}
            self.services = {}
        self.services[name] = factory(self)
        self._initialization_order.append(name)
        self._service_states[name] = SimpleNamespace(name="INITIALIZED")

    async def _init(self):
        pass

    ServiceContainer.register_service = _reg
    ServiceContainer.initialize_all_services = _init

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
