# legal_ai_system/services/__init__.py
"""Legal AI System package entry point."""

__all__ = [
    "LegalAIIntegrationService",
]

try:
    from .services.integration_service import LegalAIIntegrationService
except Exception:  # pragma: no cover - optional during minimal setups
    LegalAIIntegrationService = None
