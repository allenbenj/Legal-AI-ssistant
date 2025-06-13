import unittest

from legal_ai_system.services.service_container import ServiceContainer, register_all_agents


class DummyConfig:
    def get(self, key, default=None):
        return {}


class TestDynamicAgentRegistration(unittest.IsolatedAsyncioTestCase):
    async def test_register_all_agents_discovers_agents(self) -> None:
        container = ServiceContainer()
        await register_all_agents(container, DummyConfig())
        self.assertIn("legalreasoningengine", container._service_factories)
        self.assertTrue(container._service_factories["legalreasoningengine"]["is_async"])
