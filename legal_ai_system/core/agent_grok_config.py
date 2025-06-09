"""
Agent Grok Configuration
Configures all agents to use Grok-Mini for their analytics by default
"""

import os
from typing import Any, Dict, Optional

from .grok_config import (
    DEFAULT_GROK_MODEL,
    GROK_MODELS_CONFIG,
    LEGAL_ANALYSIS_PARAMS,
    create_grok_config,
    create_grok_legal_settings,
)
from .llm_providers import LLMConfig, LLMProviderEnum

# Force all agents to use Grok-Mini by default
AGENT_DEFAULT_MODEL = "grok-3-mini"
AGENT_FALLBACK_MODEL = "grok-2-1212"


def create_agent_grok_config(api_key: Optional[str] = None) -> LLMConfig:
    """
    Create LLM configuration for agents using Grok-Mini

    Args:
        api_key: XAI API key (will use environment variable if not provided)

    Returns:
        LLMConfig configured for Grok-Mini with agent-optimized settings
    """
    resolved_api_key = api_key or os.getenv("XAI_API_KEY")

    if not resolved_api_key:
        raise ValueError(
            "XAI API key required. Set XAI_API_KEY environment variable or provide api_key parameter."
        )

    # Get Grok-Mini configuration
    model_config = GROK_MODELS_CONFIG[AGENT_DEFAULT_MODEL]

    return LLMConfig(
        provider=LLMProviderEnum.XAI,
        model=model_config["model_name"],
        api_key=resolved_api_key,
        base_url="https://api.x.ai/v1",
        temperature=LEGAL_ANALYSIS_PARAMS[
            "legal_temperature"
        ],  # Use legal-optimized temperature
        max_tokens=model_config["max_tokens"],
        timeout=60,
        retry_attempts=3,
        retry_delay=1.0,
    )


def get_agent_llm_settings() -> Dict[str, Any]:
    """
    Get system-wide LLM settings for all agents to use Grok-Mini

    Returns:
        Dictionary of settings that configure all agents to use Grok-Mini
    """
    return {
        # Primary LLM Provider Configuration
        "llm_provider": "xai",
        "llm_model": AGENT_DEFAULT_MODEL,
        "llm_temperature": LEGAL_ANALYSIS_PARAMS["legal_temperature"],
        "llm_max_tokens": GROK_MODELS_CONFIG[AGENT_DEFAULT_MODEL]["max_tokens"],
        # XAI-specific settings
        "xai_model": AGENT_DEFAULT_MODEL,
        "xai_base_url": "https://api.x.ai/v1",
        "xai_api_key": os.getenv("XAI_API_KEY"),
        # Fallback configuration
        "fallback_provider": "xai",
        "fallback_model": AGENT_FALLBACK_MODEL,
        # Agent-specific optimizations
        "agent_temperature_override": {
            "LegalAnalysisAgent": LEGAL_ANALYSIS_PARAMS["legal_temperature"],
            "ViolationDetectorAgent": LEGAL_ANALYSIS_PARAMS["reasoning_temperature"],
            "CitationAnalysisAgent": LEGAL_ANALYSIS_PARAMS["citation_temperature"],
            "SemanticAnalysisAgent": LEGAL_ANALYSIS_PARAMS["legal_temperature"],
            "StructuralAnalysisAgent": LEGAL_ANALYSIS_PARAMS["legal_temperature"],
            "OntologyExtractionAgent": LEGAL_ANALYSIS_PARAMS["reasoning_temperature"],
            "EntityExtractionAgent": LEGAL_ANALYSIS_PARAMS["reasoning_temperature"],
            "AutoTaggingAgent": LEGAL_ANALYSIS_PARAMS["summary_temperature"],
            "NoteTakingAgent": LEGAL_ANALYSIS_PARAMS["summary_temperature"],
            "TextCorrectionAgent": LEGAL_ANALYSIS_PARAMS["citation_temperature"],
            "DocumentProcessorAgent": LEGAL_ANALYSIS_PARAMS["legal_temperature"],
        },
        # Performance settings optimized for Grok-Mini
        "max_concurrent_agents": 3,  # Conservative for rate limits
        "agent_timeout_seconds": 120,  # Longer timeout for complex analysis
        "retry_failed_agents": True,
        "max_agent_retries": 2,
        # Context management for agents
        "max_context_tokens": int(
            GROK_MODELS_CONFIG[AGENT_DEFAULT_MODEL]["context_length"] * 0.7
        ),
        "auto_summarize_threshold": int(
            GROK_MODELS_CONFIG[AGENT_DEFAULT_MODEL]["context_length"] * 0.5
        ),
        "agent_context_preservation": True,
        # Model switching rules for agents
        "enable_model_switching": True,
        "complexity_model_mapping": {
            "simple": AGENT_DEFAULT_MODEL,
            "moderate": AGENT_DEFAULT_MODEL,
            "complex": "grok-3-reasoning",  # Use reasoning model for complex tasks
            "expert": "grok-3-reasoning",
        },
    }


def configure_agent_service_container(
    service_container: Any, api_key: Optional[str] = None
) -> None:
    """
    Configure the service container to provide Grok-Mini LLM config to all agents

    Args:
        service_container: The service container instance
        api_key: XAI API key (optional, will use environment variable)
    """
    try:
        # Create Grok configuration for agents
        agent_llm_config = create_agent_grok_config(api_key)

        # Register LLM configuration in service container
        if hasattr(service_container, "register_service"):
            service_container.register_service("llm_config", agent_llm_config)
            service_container.register_service("agent_llm_config", agent_llm_config)

        # Update LLM manager if it exists
        if hasattr(service_container, "get_service"):
            llm_manager = service_container.get_service("llm_manager")
            if llm_manager and hasattr(llm_manager, "update_config"):
                llm_manager.update_config(agent_llm_config)

        print(f"✅ Configured all agents to use {AGENT_DEFAULT_MODEL}")

    except Exception as e:
        print(f"❌ Failed to configure agents for Grok: {e}")
        raise


class AgentGrokManager:
    """
    Manager class to ensure all agents use Grok-Mini consistently
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.agent_configs = {}

        if not self.api_key:
            raise ValueError("XAI API key required for agent Grok configuration")

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get Grok configuration for a specific agent"""
        if agent_name not in self.agent_configs:
            base_settings = get_agent_llm_settings()

            # Agent-specific temperature if configured
            temp_overrides = base_settings.get("agent_temperature_override", {})
            if agent_name in temp_overrides:
                base_settings["llm_temperature"] = temp_overrides[agent_name]

            self.agent_configs[agent_name] = base_settings

        return self.agent_configs[agent_name]

    def create_llm_config_for_agent(self, agent_name: str) -> LLMConfig:
        """Create LLMConfig instance for specific agent"""
        agent_config = self.get_agent_config(agent_name)

        return LLMConfig(
            provider=LLMProviderEnum.XAI,
            model=agent_config["llm_model"],
            api_key=self.api_key,
            base_url=agent_config["xai_base_url"],
            temperature=agent_config["llm_temperature"],
            max_tokens=agent_config["llm_max_tokens"],
            timeout=agent_config.get("agent_timeout_seconds", 120),
            retry_attempts=agent_config.get("max_agent_retries", 2),
            retry_delay=1.0,
        )

    def validate_agent_grok_setup(self) -> Dict[str, Any]:
        """Validate that agents are properly configured for Grok"""
        validation = {
            "api_key_configured": bool(self.api_key),
            "default_model": AGENT_DEFAULT_MODEL,
            "fallback_model": AGENT_FALLBACK_MODEL,
            "models_available": list(GROK_MODELS_CONFIG.keys()),
            "agent_specific_configs": len(self.agent_configs),
            "ready": False,
        }

        validation["ready"] = validation["api_key_configured"]

        return validation


# Global manager instance
_agent_grok_manager = None


def get_agent_grok_manager(api_key: Optional[str] = None) -> AgentGrokManager:
    """Get or create the global agent Grok manager"""
    global _agent_grok_manager

    if _agent_grok_manager is None:
        _agent_grok_manager = AgentGrokManager(api_key)

    return _agent_grok_manager


def ensure_agents_use_grok(
    service_container: Any = None, api_key: Optional[str] = None
) -> bool:
    """
    Ensure all agents are configured to use Grok-Mini

    Args:
        service_container: Optional service container to configure
        api_key: Optional XAI API key

    Returns:
        True if successfully configured, False otherwise
    """
    try:
        # Initialize the agent Grok manager
        manager = get_agent_grok_manager(api_key)

        # Configure service container if provided
        if service_container:
            configure_agent_service_container(service_container, api_key)

        # Validate setup
        validation = manager.validate_agent_grok_setup()

        if validation["ready"]:
            print(f"✅ All agents configured to use {AGENT_DEFAULT_MODEL}")
            print(f"📊 Available models: {validation['models_available']}")
            print(f"🔄 Fallback model: {AGENT_FALLBACK_MODEL}")
            return True
        else:
            print("❌ Agent Grok configuration incomplete")
            print(f"   API Key: {'✅' if validation['api_key_configured'] else '❌'}")
            return False

    except Exception as e:
        print(f"❌ Failed to configure agents for Grok: {e}")
        return False


# Environment setup template for agent Grok configuration
AGENT_GROK_ENV_TEMPLATE = f"""
# Agent Grok Configuration
# All agents will use these settings for LLM operations

# Primary XAI Configuration
XAI_API_KEY=your_xai_api_key_here
LLM_PROVIDER=xai
LLM_MODEL={AGENT_DEFAULT_MODEL}
LLM_TEMPERATURE={LEGAL_ANALYSIS_PARAMS["legal_temperature"]}
LLM_MAX_TOKENS={GROK_MODELS_CONFIG[AGENT_DEFAULT_MODEL]["max_tokens"]}

# Agent Performance Settings
MAX_CONCURRENT_AGENTS=3
AGENT_TIMEOUT_SECONDS=120
ENABLE_MODEL_SWITCHING=true

# Context Management
MAX_CONTEXT_TOKENS={int(GROK_MODELS_CONFIG[AGENT_DEFAULT_MODEL]["context_length"] * 0.7)}
AUTO_SUMMARIZE_THRESHOLD={int(GROK_MODELS_CONFIG[AGENT_DEFAULT_MODEL]["context_length"] * 0.5)}

# Fallback Configuration
FALLBACK_PROVIDER=xai
FALLBACK_MODEL={AGENT_FALLBACK_MODEL}
"""


def print_agent_grok_setup_instructions():
    """Print setup instructions for agent Grok configuration"""
    print("\n" + "=" * 60)
    print("AGENT GROK-MINI CONFIGURATION INSTRUCTIONS")
    print("=" * 60)

    print("\n1. SET ENVIRONMENT VARIABLES:")
    print("   export XAI_API_KEY='your_xai_api_key_here'")
    print(f"   export LLM_PROVIDER=xai")
    print(f"   export LLM_MODEL={AGENT_DEFAULT_MODEL}")

    print("\n2. IMPORT AND CONFIGURE IN YOUR APPLICATION:")
    print(
        "   from legal_ai_system.core.agent_grok_config import ensure_agents_use_grok"
    )
    print("   ensure_agents_use_grok(service_container, api_key)")

    print("\n3. VERIFY CONFIGURATION:")
    print(
        "   from legal_ai_system.core.agent_grok_config import get_agent_grok_manager"
    )
    print("   manager = get_agent_grok_manager()")
    print("   validation = manager.validate_agent_grok_setup()")
    print("   print(validation)")

    print(f"\n✅ ALL AGENTS WILL USE: {AGENT_DEFAULT_MODEL}")
    print(f"🔄 FALLBACK MODEL: {AGENT_FALLBACK_MODEL}")
    print(f"🎯 OPTIMIZED FOR: Legal document analysis")
    print(f"⚡ PERFORMANCE: Fast, efficient processing")
    print(f"💰 COST: Most economical Grok model")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_agent_grok_setup_instructions()
