import asyncio
import pytest


class ServiceContainer:
    def __init__(self) -> None:
        self._factories = []
        self._initialization_order = []
        self._service_states = {}

    async def register_service(self, name: str, factory):
        self._factories.append((name, factory))
        self._service_states[name] = type("State", (), {"name": "REGISTERED"})

    async def initialize_all_services(self):
        for name, factory in self._factories:
            factory(self)
            self._initialization_order.append(name)
            self._service_states[name] = type("State", (), {"name": "INITIALIZED"})

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
