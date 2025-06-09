#!/usr/bin/env python3
"""External Service Connectivity Tests

This script checks basic connectivity to configured LLM providers,
embedding services, and the document processor agent. It is intended
for quick diagnostics and does not replace full integration tests.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path so tests can run from any location
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from legal_ai_system.core.llm_providers import (
    LLMConfig,
    LLMProviderEnum,
    LLMManager,
)
from legal_ai_system.core.embedding_manager import EmbeddingManager


async def test_llm_provider() -> bool:
    """Verify that the primary LLM provider can be initialized."""
    print("\n🔌 Testing LLM provider connectivity...")
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider in {"openai", "xai"}:
        api_key_env = "OPENAI_API_KEY" if provider == "openai" else "XAI_API_KEY"
        api_key = os.getenv(api_key_env)
        base_url = os.getenv(
            "OPENAI_BASE_URL" if provider == "openai" else "XAI_BASE_URL",
            None,
        )
        model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        if not api_key:
            print(f"  ⚠️  {api_key_env} not set. Skipping LLM provider test.")
            return False
        config = LLMConfig(
            provider=LLMProviderEnum.OPENAI if provider == "openai" else LLMProviderEnum.XAI,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
    elif provider == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model = os.getenv("LLM_MODEL", "llama3")
        config = LLMConfig(
            provider=LLMProviderEnum.OLLAMA,
            model=model,
            base_url=host,
        )
    else:
        print(f"  ⚠️  Unknown provider '{provider}'. Skipping.")
        return False

    manager = LLMManager(primary_config=config)
    try:
        await manager.initialize()
        health = await manager.health_check()
        print(f"  ✓ LLM provider status: {health.get('overall_status')}")
        await manager.shutdown()
        return True
    except Exception as exc:
        print(f"  ✗ LLM provider check failed: {exc}")
        return False


async def test_embedding_service() -> bool:
    """Check that the embedding service can initialize."""
    print("\n🧠 Testing embedding service connectivity...")
    manager = EmbeddingManager()
    try:
        await manager.initialize()
        health = await manager.health_check()
        print(f"  ✓ Embedding service status: {health.get('status')}")
        await manager.shutdown()
        return health.get("status") == "healthy"
    except Exception as exc:
        print(f"  ✗ Embedding service initialization failed: {exc}")
        return False


async def test_document_processor() -> bool:
    """Run the document processor agent health check."""
    print("\n📄 Testing document processor agent...")
    try:
        from legal_ai_system.agents.document_processor_agent import DocumentProcessorAgent
    except Exception as exc:  # Missing optional deps
        print(f"  ⚠️  DocumentProcessorAgent import failed: {exc}")
        return False

    agent = DocumentProcessorAgent(None)
    try:
        health = await agent.health_check()
        print(f"  ✓ Document processor status: {health.get('status')}")
        return True
    except Exception as exc:
        print(f"  ✗ Document processor health check failed: {exc}")
        return False


async def main() -> None:
    results = await asyncio.gather(
        test_llm_provider(),
        test_embedding_service(),
        test_document_processor(),
    )

    passed = sum(1 for r in results if r)
    print(f"\nExternal Service Tests Passed: {passed}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
