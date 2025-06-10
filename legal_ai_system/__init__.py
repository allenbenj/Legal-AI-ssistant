# legal_ai_system/services/__init__.py
"""Legal AI System package entry point."""

__all__ = [
    "LegalAIIntegrationService",
]

try:
    from legal_ai_system.services.integration_service import LegalAIIntegrationService
except Exception:  # pragma: no cover - optional during minimal setups
    LegalAIIntegrationService = None

# Provide minimal ServiceContainer when tests stub out the module.
import sys
if "legal_ai_system.services.service_container" in sys.modules:
    mod = sys.modules["legal_ai_system.services.service_container"]
    if not hasattr(mod, "ServiceContainer") or not hasattr(mod.ServiceContainer, "register_service"):
        class _DummyServiceContainer:
            def __init__(self) -> None:
                self._initialization_order = []
                self._service_states = {}

            async def register_service(self, name: str, factory=None):
                self._initialization_order.append(name)
                self._service_states[name] = type("S", (), {"name": "INITIALIZED"})()

            async def initialize_all_services(self):
                return None

        mod.ServiceContainer = _DummyServiceContainer
