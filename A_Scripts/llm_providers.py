"""
Enhanced LLM Provider System with Multi-Provider Support
Implements the strategy pattern for different LLM providers
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
import time
from enum import Enum

# Provider-specific imports with fallbacks
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    XAI = "xai"

class LLMConfig(BaseModel):
    """Configuration for LLM providers"""
    provider: LLMProvider = Field(LLMProvider.OLLAMA, description="LLM provider")
    model: str = Field("llama3.2", description="Model name")
    api_key: Optional[str] = Field(None, description="API key if required")
    base_url: Optional[str] = Field(None, description="Base URL for API")
    temperature: float = Field(0.7, description="Temperature for generation", ge=0.0, le=2.0)
    max_tokens: int = Field(4096, description="Maximum tokens", gt=0)
    timeout: int = Field(60, description="Request timeout in seconds", gt=0)
    retry_attempts: int = Field(3, description="Number of retry attempts", ge=0)
    retry_delay: float = Field(1.0, description="Delay between retries", ge=0.0)

class LLMResponse(BaseModel):
    """Standardized LLM response"""
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    response_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class LLMError(Exception):
    """Base exception for LLM operations"""
    def __init__(self, message: str, provider: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.name = config.provider.value
        self._client = None
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider"""
        pass
    
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion for prompt"""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check if provider is healthy"""
        pass
    
    async def shutdown(self) -> None:
        """Cleanup resources"""
        self._initialized = False

class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider for local models"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not OLLAMA_AVAILABLE:
            raise LLMError("Ollama library not available", self.name)
    
    async def initialize(self) -> None:
        """Initialize Ollama client"""
        try:
            if self.config.base_url:
                self._client = ollama.AsyncClient(host=self.config.base_url)
            else:
                self._client = ollama.AsyncClient()
            
            # Test connection
            await self._client.list()
            self._initialized = True
            logger.info(f"Ollama provider initialized: {self.config.base_url or 'default host'}")
            
        except Exception as e:
            raise LLMError(f"Failed to initialize Ollama: {e}", self.name, e)
    
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using Ollama"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        # Merge kwargs with config
        options = {
            'temperature': kwargs.get('temperature', self.config.temperature),
            'num_predict': kwargs.get('max_tokens', self.config.max_tokens),
        }
        
        for attempt in range(self.config.retry_attempts + 1):
            try:
                response = await self._client.generate(
                    model=self.config.model,
                    prompt=prompt,
                    options=options,
                    stream=False
                )
                
                response_time = time.time() - start_time
                
                return LLMResponse(
                    content=response['response'],
                    provider=self.name,
                    model=self.config.model,
                    tokens_used=response.get('eval_count'),
                    response_time=response_time,
                    metadata={
                        'total_duration': response.get('total_duration'),
                        'load_duration': response.get('load_duration'),
                        'prompt_eval_count': response.get('prompt_eval_count'),
                        'eval_count': response.get('eval_count')
                    }
                )
                
            except Exception as e:
                if attempt < self.config.retry_attempts:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise LLMError(f"Ollama completion failed: {e}", self.name, e)
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding using Ollama"""
        if not self._initialized:
            await self.initialize()
        
        try:
            response = await self._client.embeddings(
                model='nomic-embed-text',  # Default embedding model
                prompt=text
            )
            return response['embedding']
            
        except Exception as e:
            raise LLMError(f"Ollama embedding failed: {e}", self.name, e)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama health"""
        try:
            models = await self._client.list()
            return {
                "status": "healthy",
                "available_models": [m['name'] for m in models.get('models', [])],
                "configured_model": self.config.model
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

class OpenAIProvider(BaseLLMProvider):
    """OpenAI/xAI provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise LLMError("OpenAI library not available", self.name)
        
        if not config.api_key:
            raise LLMError("API key required for OpenAI provider", self.name)
    
    async def initialize(self) -> None:
        """Initialize OpenAI client"""
        try:
            client_kwargs = {
                'api_key': self.config.api_key,
                'timeout': self.config.timeout
            }
            
            if self.config.base_url:
                client_kwargs['base_url'] = self.config.base_url
            
            self._client = openai.AsyncOpenAI(**client_kwargs)
            self._initialized = True
            
            logger.info(f"OpenAI provider initialized: {self.config.base_url or 'default'}")
            
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenAI: {e}", self.name, e)
    
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using OpenAI API"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(self.config.retry_attempts + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=kwargs.get('temperature', self.config.temperature),
                    max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                    timeout=self.config.timeout
                )
                
                response_time = time.time() - start_time
                
                return LLMResponse(
                    content=response.choices[0].message.content,
                    provider=self.name,
                    model=self.config.model,
                    tokens_used=response.usage.total_tokens if response.usage else None,
                    response_time=response_time,
                    metadata={
                        'prompt_tokens': response.usage.prompt_tokens if response.usage else None,
                        'completion_tokens': response.usage.completion_tokens if response.usage else None,
                        'finish_reason': response.choices[0].finish_reason
                    }
                )
                
            except Exception as e:
                if attempt < self.config.retry_attempts:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                raise LLMError(f"OpenAI completion failed: {e}", self.name, e)
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        if not self._initialized:
            await self.initialize()
        
        try:
            response = await self._client.embeddings.create(
                model="text-embedding-ada-002",  # Default embedding model
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            raise LLMError(f"OpenAI embedding failed: {e}", self.name, e)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI health"""
        try:
            # Simple test request
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                timeout=5
            )
            return {
                "status": "healthy",
                "model": self.config.model
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

class LLMManager:
    """Manages multiple LLM providers with automatic fallback"""
    
    def __init__(self, primary_config: LLMConfig, fallback_configs: Optional[List[LLMConfig]] = None):
        self.primary_config = primary_config
        self.fallback_configs = fallback_configs or []
        
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.primary_provider: Optional[BaseLLMProvider] = None
        self.fallback_providers: List[BaseLLMProvider] = []
        
        self._call_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'provider_usage': {}
        }
    
    async def initialize(self) -> None:
        """Initialize all providers"""
        try:
            # Initialize primary provider
            self.primary_provider = self._create_provider(self.primary_config)
            await self.primary_provider.initialize()
            self.providers[self.primary_config.provider.value] = self.primary_provider
            
            # Initialize fallback providers
            for config in self.fallback_configs:
                try:
                    provider = self._create_provider(config)
                    await provider.initialize()
                    self.providers[config.provider.value] = provider
                    self.fallback_providers.append(provider)
                except Exception as e:
                    logger.warning(f"Failed to initialize fallback provider {config.provider}: {e}")
            
            logger.info(f"LLM Manager initialized with {len(self.providers)} providers")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM Manager: {e}")
            raise
    
    def _create_provider(self, config: LLMConfig) -> BaseLLMProvider:
        """Create provider instance based on config"""
        if config.provider == LLMProvider.OLLAMA:
            return OllamaProvider(config)
        elif config.provider in [LLMProvider.OPENAI, LLMProvider.XAI]:
            return OpenAIProvider(config)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")
    
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion with automatic fallback"""
        self._call_stats['total_calls'] += 1
        
        # Try primary provider first
        if self.primary_provider:
            try:
                response = await self.primary_provider.complete(prompt, **kwargs)
                self._call_stats['successful_calls'] += 1
                self._update_provider_stats(self.primary_provider.name, success=True)
                return response
            except Exception as e:
                logger.warning(f"Primary provider {self.primary_provider.name} failed: {e}")
                self._update_provider_stats(self.primary_provider.name, success=False)
        
        # Try fallback providers
        for provider in self.fallback_providers:
            try:
                logger.info(f"Trying fallback provider: {provider.name}")
                response = await provider.complete(prompt, **kwargs)
                self._call_stats['successful_calls'] += 1
                self._update_provider_stats(provider.name, success=True)
                return response
            except Exception as e:
                logger.warning(f"Fallback provider {provider.name} failed: {e}")
                self._update_provider_stats(provider.name, success=False)
                continue
        
        # All providers failed
        self._call_stats['failed_calls'] += 1
        raise LLMError("All LLM providers failed", "all")
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding with fallback"""
        # Try primary provider first
        if self.primary_provider:
            try:
                return await self.primary_provider.embed(text)
            except Exception as e:
                logger.warning(f"Primary provider embedding failed: {e}")
        
        # Try fallback providers
        for provider in self.fallback_providers:
            try:
                return await provider.embed(text)
            except Exception as e:
                logger.warning(f"Fallback provider {provider.name} embedding failed: {e}")
                continue
        
        raise LLMError("All providers failed for embedding", "all")
    
    def _update_provider_stats(self, provider_name: str, success: bool) -> None:
        """Update provider usage statistics"""
        if provider_name not in self._call_stats['provider_usage']:
            self._call_stats['provider_usage'][provider_name] = {
                'calls': 0,
                'successes': 0,
                'failures': 0
            }
        
        stats = self._call_stats['provider_usage'][provider_name]
        stats['calls'] += 1
        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all providers"""
        health = {
            "status": "healthy",
            "primary_provider": self.primary_config.provider.value,
            "providers": {},
            "stats": self._call_stats
        }
        
        for name, provider in self.providers.items():
            try:
                provider_health = await provider.health_check()
                health["providers"][name] = provider_health
                if provider_health.get("status") != "healthy":
                    health["status"] = "degraded"
            except Exception as e:
                health["providers"][name] = {"status": "error", "error": str(e)}
                health["status"] = "degraded"
        
        return health
    
    async def shutdown(self) -> None:
        """Shutdown all providers"""
        for provider in self.providers.values():
            try:
                await provider.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down provider: {e}")
        
        self.providers.clear()
        self.primary_provider = None
        self.fallback_providers.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self._call_stats.copy()