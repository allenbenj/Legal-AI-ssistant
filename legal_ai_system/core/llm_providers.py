# legal_ai_system/core/llm_providers.py

#Enhanced LLM Provider System with Multi-Provider Support
#Implements the strategy pattern for different LLM providers.


import asyncio
import time # Replaced logging with detailed_logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field as PydanticField # Alias to avoid conflict with FastAPI's Field
from enum import Enum

# Use detailed_logging
from .detailed_logging import get_detailed_logger, LogCategory, detailed_log_function # Assuming in core
# Import exceptions from the core module
from .unified_exceptions import LLMProviderError # Assuming in core

# Get a logger for this module
llm_provider_logger = get_detailed_logger("LLMProviders", LogCategory.LLM)


class LLMProviderEnum(str, Enum): # Renamed to avoid conflict, and made it a str Enum for FastAPI
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    XAI = "xai" # Added XAI for Grok

class LLMConfig(BaseModel):
    """Configuration for LLM providers"""
    provider: LLMProviderEnum = PydanticField(LLMProviderEnum.OLLAMA, description="LLM provider")
    model: str = PydanticField("llama3.2", description="Model name") # Adjusted default for Ollama
    api_key: Optional[str] = PydanticField(None, description="API key if required")
    base_url: Optional[str] = PydanticField(None, description="Base URL for API")
    temperature: float = PydanticField(0.7, description="Temperature for generation", ge=0.0, le=2.0)
    max_tokens: int = PydanticField(4096, description="Maximum tokens for response", gt=0) # Clarified
    timeout: int = PydanticField(60, description="Request timeout in seconds", gt=0)
    retry_attempts: int = PydanticField(3, description="Number of retry attempts", ge=0)
    retry_delay: float = PydanticField(1.0, description="Delay between retries in seconds", ge=0.0) # Clarified unit

class LLMResponse(BaseModel):
    """Standardized LLM response"""
    content: str
    provider_name: str # Renamed from provider to avoid conflict with LLMConfig.provider
    model_name: str    # Renamed from model
    tokens_used: Optional[int] = None
    response_time_sec: float # Renamed for clarity
    metadata: Dict[str, Any] = PydanticField(default_factory=dict)

# LLMError is now imported from unified_exceptions.py

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.name = config.provider.value # This is the string value of the enum
        self._client: Optional[Any] = None # More generic type hint
        self._initialized = False
        self.logger = get_detailed_logger(f"LLMProvider_{self.name.upper()}", LogCategory.LLM)
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider"""
        self.logger.info(f"Initializing provider.", parameters={'provider_name': self.name})
        pass
    
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion for prompt"""
        pass
    
    @abstractmethod
    async def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding for text."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check if provider is healthy"""
        pass
    
    async def shutdown(self) -> None:
        """Cleanup resources"""
        self.logger.info(f"Shutting down provider.", parameters={'provider_name': self.name})
        self._initialized = False
        self._client = None # Explicitly clear client

class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider for local models"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            global ollama # Make sure we're using the global ollama if available
            import ollama
            self.ollama_sdk = ollama # Store the SDK
        except ImportError:
            raise LLMProviderError("Ollama SDK not available. Please install with `pip install ollama`.", self.name)

    @detailed_log_function(LogCategory.LLM)
    async def initialize(self) -> None:
        """Initialize Ollama client"""
        if self._initialized: return
        try:
            host = self.config.base_url or None # ollama.AsyncClient handles default host if None
            self._client = self.ollama_sdk.AsyncClient(host=host, timeout=self.config.timeout)
            
            await self._client.list() # Test connection
            self._initialized = True
            self.logger.info(f"Ollama provider initialized.", parameters={'host': host or 'default'})
            
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Ollama: {str(e)}", self.name, original_error=e)

    @detailed_log_function(LogCategory.LLM)
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using Ollama"""
        if not self._initialized or not self._client:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        options = {
            'temperature': kwargs.get('temperature', self.config.temperature),
            'num_predict': kwargs.get('max_tokens', self.config.max_tokens),
            # Add other Ollama specific options if needed
        }
        
        for attempt in range(self.config.retry_attempts + 1):
            try:
                self.logger.debug(f"Ollama completion attempt {attempt + 1}", parameters={'model': self.config.model})
                response = await self._client.generate( # type: ignore
                    model=self.config.model,
                    prompt=prompt,
                    options=options,
                    stream=False # Assuming non-streaming for now
                )
                
                response_time = time.perf_counter() - start_time
                
                return LLMResponse(
                    content=response['response'],
                    provider_name=self.name,
                    model_name=self.config.model, # Model used for this specific call
                    tokens_used=response.get('eval_count'), # Example, check Ollama docs for actual token counts
                    response_time_sec=response_time,
                    metadata={
                        'total_duration_ns': response.get('total_duration'),
                        'load_duration_ns': response.get('load_duration'),
                        'prompt_eval_count': response.get('prompt_eval_count'),
                        'eval_count': response.get('eval_count')
                    }
                )
            except Exception as e:
                self.logger.warning(f"Ollama completion attempt {attempt + 1} failed.", 
                                   parameters={'error': str(e)}, exception=e)
                if attempt < self.config.retry_attempts:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1)) # Exponential backoff
                else:
                    raise LLMProviderError(f"Ollama completion failed after {attempt+1} attempts: {str(e)}", self.name, original_error=e)
        raise LLMProviderError("Ollama completion failed: Unknown error after retries", self.name) # Should not be reached

    @detailed_log_function(LogCategory.LLM)
    async def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding using Ollama"""
        if not self._initialized or not self._client:
            await self.initialize()
        
        embedding_model = model or "nomic-embed-text" # Common Ollama embedding model
        self.logger.debug(f"Generating Ollama embedding", parameters={'model': embedding_model})
        try:
            response = await self._client.embeddings( # type: ignore
                model=embedding_model,
                prompt=text
            )
            return response['embedding']
            
        except Exception as e:
            raise LLMProviderError(f"Ollama embedding failed: {str(e)}", self.name, original_error=e)
    
    @detailed_log_function(LogCategory.LLM)
    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama health"""
        if not self._client: # Not initialized yet
            return {"status": "uninitialized", "provider": self.name}
        try:
            models_info = await self._client.list() # type: ignore
            return {
                "status": "healthy",
                "provider": self.name,
                "available_models": [m['name'] for m in models_info.get('models', [])],
                "configured_model": self.config.model,
                "host": self.config.base_url or 'default'
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.name,
                "error": str(e)
            }

class OpenAICompatibleProvider(BaseLLMProvider): # For OpenAI, XAI, and other OpenAI-like APIs
    """Provider for OpenAI and OpenAI-compatible APIs like xAI Grok."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            global openai # Make sure we're using the global openai if available
            import openai
            self.openai_sdk = openai # Store the SDK
        except ImportError:
            raise LLMProviderError(f"{self.name} SDK (openai) not available. Please install with `pip install openai`.", self.name)
        
        if not config.api_key:
            raise LLMProviderError(f"API key required for {self.name} provider", self.name)
    
    @detailed_log_function(LogCategory.LLM)
    async def initialize(self) -> None:
        """Initialize OpenAI/XAI client"""
        if self._initialized: return
        try:
            client_kwargs: Dict[str, Any] = { # type: ignore[no-redef]
                'api_key': self.config.api_key,
                'timeout': self.config.timeout, # OpenAI timeout is a httpx.Timeout object or float
            }
            
            if self.config.base_url:
                client_kwargs['base_url'] = self.config.base_url
            
            self._client = self.openai_sdk.AsyncOpenAI(**client_kwargs)
            self._initialized = True
            self.logger.info(f"{self.name} provider initialized.", parameters={'base_url': self.config.base_url or 'default OpenAI'})
            
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize {self.name}: {str(e)}", self.name, original_error=e)

    @detailed_log_function(LogCategory.LLM)
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using OpenAI/XAI API"""
        if not self._initialized or not self._client:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        messages = [{"role": "user", "content": prompt}] # Basic chat format
        
        for attempt in range(self.config.retry_attempts + 1):
            try:
                self.logger.debug(f"{self.name} completion attempt {attempt + 1}", parameters={'model': self.config.model})
                response = await self._client.chat.completions.create( # type: ignore
                    model=self.config.model,
                    messages=messages, # type: ignore
                    temperature=kwargs.get('temperature', self.config.temperature),
                    max_tokens=kwargs.get('max_tokens', self.config.max_tokens)
                    # timeout is part of client config
                )
                
                response_time = time.perf_counter() - start_time
                
                content = response.choices[0].message.content if response.choices and response.choices[0].message else ""
                usage = response.usage

                return LLMResponse(
                    content=content or "",
                    provider_name=self.name,
                    model_name=self.config.model, # Or response.model if available and preferred
                    tokens_used=usage.total_tokens if usage else None,
                    response_time_sec=response_time,
                    metadata={
                        'prompt_tokens': usage.prompt_tokens if usage else None,
                        'completion_tokens': usage.completion_tokens if usage else None,
                        'finish_reason': response.choices[0].finish_reason if response.choices else None,
                        'response_id': response.id if hasattr(response, 'id') else None
                    }
                )
            except Exception as e:
                self.logger.warning(f"{self.name} completion attempt {attempt + 1} failed.", 
                                   parameters={'error': str(e)}, exception=e)
                if attempt < self.config.retry_attempts:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1)) # Exponential backoff
                else:
                    raise LLMProviderError(f"{self.name} completion failed after {attempt+1} attempts: {str(e)}", self.name, original_error=e)
        raise LLMProviderError(f"{self.name} completion failed: Unknown error after retries", self.name) # Should not be reached

    @detailed_log_function(LogCategory.LLM)
    async def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding using OpenAI/XAI API"""
        if not self._initialized or not self._client:
            await self.initialize()
        
        # Standard OpenAI embedding model, xAI might have a different one or require model in config
        embedding_model = model or ("text-embedding-ada-002" if self.config.provider == LLMProviderEnum.OPENAI else self.config.model)
        self.logger.debug(f"Generating {self.name} embedding", parameters={'model': embedding_model})
        try:
            response = await self._client.embeddings.create( # type: ignore
                model=embedding_model,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            raise LLMProviderError(f"{self.name} embedding failed: {str(e)}", self.name, original_error=e)

    @detailed_log_function(LogCategory.LLM)
    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI/XAI health"""
        if not self._client: # Not initialized yet
            return {"status": "uninitialized", "provider": self.name}
        try:
            # Simple test: list models if API supports it, or a tiny completion
            # For OpenAI compatible, listing models might not be standard.
            # A small completion is a better health check.
            await self._client.chat.completions.create( # type: ignore
                model=self.config.model,
                messages=[{"role": "user", "content": "Health check"}], # type: ignore
                max_tokens=1,
                temperature=0.1
            )
            return {
                "status": "healthy",
                "provider": self.name,
                "configured_model": self.config.model,
                "base_url": self.config.base_url or 'default'
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.name,
                "error": str(e)
            }

class LLMManager:
    """Manages multiple LLM providers with automatic fallback and detailed logging."""
    
    @detailed_log_function(LogCategory.LLM)
    def __init__(self, primary_config: LLMConfig, fallback_configs: Optional[List[LLMConfig]] = None):
        self.primary_config = primary_config
        self.fallback_configs = fallback_configs or []
        self.logger = get_detailed_logger("LLMManager", LogCategory.LLM)
        
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.primary_provider: Optional[BaseLLMProvider] = None
        self.fallback_providers: List[BaseLLMProvider] = []
        
        self._call_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'provider_usage': {} # provider_name: {'calls': N, 'successes': M, 'failures': K}
        }
        self.logger.info("LLMManager initialized", parameters={'primary_provider': primary_config.provider.value})

    @detailed_log_function(LogCategory.LLM)
    async def initialize(self) -> None:
        """Initialize all configured LLM providers."""
        self.logger.info("Initializing LLM providers...")
        try:
            self.primary_provider = self._create_provider(self.primary_config)
            await self.primary_provider.initialize()
            self.providers[self.primary_provider.name] = self.primary_provider
            self.logger.info(f"Primary provider '{self.primary_provider.name}' initialized.")

            for fb_config in self.fallback_configs:
                try:
                    provider = self._create_provider(fb_config)
                    await provider.initialize()
                    self.providers[provider.name] = provider
                    self.fallback_providers.append(provider)
                    self.logger.info(f"Fallback provider '{provider.name}' initialized.")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize fallback provider '{fb_config.provider.value}'.", 
                                       parameters={'error': str(e)}, exception=e)
            
            self.logger.info(f"LLM providers initialization complete. Total providers: {len(self.providers)}")
            
        except Exception as e:
            self.logger.error(f"LLMManager initialization failed.", exception=e)
            raise LLMProviderError("LLMManager failed to initialize.", "manager", original_error=e)

    def _create_provider(self, config: LLMConfig) -> BaseLLMProvider:
        """Factory method to create provider instances."""
        self.logger.debug(f"Creating provider instance", parameters={'provider': config.provider.value, 'model': config.model})
        if config.provider == LLMProviderEnum.OLLAMA:
            return OllamaProvider(config)
        elif config.provider == LLMProviderEnum.OPENAI:
            return OpenAICompatibleProvider(config) # Using the compatible provider for OpenAI
        elif config.provider == LLMProviderEnum.XAI:
            return OpenAICompatibleProvider(config) # XAI Grok uses OpenAI-compatible API
        else:
            msg = f"Unsupported LLM provider type: {config.provider.value}"
            self.logger.error(msg)
            raise ValueError(msg)
    
    @detailed_log_function(LogCategory.LLM)
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion with automatic fallback."""
        self._call_stats['total_calls'] += 1
        self.logger.info("Attempting completion via primary provider.", parameters={'prompt_start': prompt[:50]+"..."})

        if self.primary_provider:
            try:
                response = await self.primary_provider.complete(prompt, **kwargs)
                self._call_stats['successful_calls'] += 1
                self._update_provider_stats(self.primary_provider.name, success=True)
                return response
            except Exception as e:
                self.logger.warning(f"Primary provider '{self.primary_provider.name}' failed.", 
                                   parameters={'error': str(e)}, exception=e)
                self._update_provider_stats(self.primary_provider.name, success=False)
        
        self.logger.info("Primary provider failed, attempting fallback providers.")
        for provider in self.fallback_providers:
            try:
                self.logger.info(f"Trying fallback provider: {provider.name}")
                response = await provider.complete(prompt, **kwargs)
                self._call_stats['successful_calls'] += 1
                self._update_provider_stats(provider.name, success=True)
                return response
            except Exception as e:
                self.logger.warning(f"Fallback provider '{provider.name}' failed.", 
                                   parameters={'error': str(e)}, exception=e)
                self._update_provider_stats(provider.name, success=False)
                continue
        
        self._call_stats['failed_calls'] += 1
        self.logger.error("All LLM providers failed for completion task.")
        raise LLMProviderError("All LLM providers failed for completion.", "all_providers")

    @detailed_log_function(LogCategory.LLM)
    async def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding with fallback."""
        self.logger.info("Attempting embedding via primary provider.", parameters={'text_start': text[:50]+"..."})
        if self.primary_provider:
            try:
                return await self.primary_provider.embed(text, model=model)
            except Exception as e:
                self.logger.warning(f"Primary provider embedding failed.", parameters={'error': str(e)}, exception=e)
        
        self.logger.info("Primary provider embedding failed, attempting fallback providers.")
        for provider in self.fallback_providers:
            try:
                self.logger.info(f"Trying fallback provider for embedding: {provider.name}")
                return await provider.embed(text, model=model) # Pass model if specified
            except Exception as e:
                self.logger.warning(f"Fallback provider '{provider.name}' embedding failed.", 
                                   parameters={'error': str(e)}, exception=e)
                continue
        
        self.logger.error("All LLM providers failed for embedding task.")
        raise LLMProviderError("All providers failed for embedding.", "all_providers")
    
    def _update_provider_stats(self, provider_name: str, success: bool) -> None:
        """Update provider usage statistics."""
        if provider_name not in self._call_stats['provider_usage']:
            self._call_stats['provider_usage'][provider_name] = {'calls': 0, 'successes': 0, 'failures': 0}
        
        stats = self._call_stats['provider_usage'][provider_name]
        stats['calls'] += 1
        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1
        self.logger.trace("Provider stats updated", parameters={'provider': provider_name, 'stats': stats})

    @detailed_log_function(LogCategory.LLM)
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all managed LLM providers."""
        self.logger.info("Performing LLMManager health check.")
        health_summary = {
            "overall_status": "healthy", # Assume healthy initially
            "primary_provider_configured": self.primary_config.provider.value if self.primary_config else "None",
            "providers_status": {},
            "usage_statistics": self._call_stats.copy()
        }
        
        for name, provider_instance in self.providers.items(): # Iterate over initialized providers
            try:
                provider_health = await provider_instance.health_check()
                health_summary["providers_status"][name] = provider_health
                if provider_health.get("status") != "healthy":
                    health_summary["overall_status"] = "degraded"
            except Exception as e:
                health_summary["providers_status"][name] = {"status": "error", "error_message": str(e)}
                health_summary["overall_status"] = "degraded" # If any provider errors, manager is degraded
        
        if not self.providers:
            health_summary["overall_status"] = "error"  # No providers initialized
            health_summary["providers_status"]["manager"] = {
                "status": "error",
                "error_message": "No LLM providers initialized."
            }


        self.logger.info("LLMManager health check complete.", parameters={'overall_status': health_summary["overall_status"]})
        return health_summary
    
    @detailed_log_function(LogCategory.LLM)
    async def shutdown(self) -> None:
        """Shutdown all managed LLM providers."""
        self.logger.info("Shutting down LLMManager and all providers.")
        for name, provider_instance in self.providers.items():
            try:
                await provider_instance.shutdown()
                self.logger.info(f"Provider '{name}' shut down.")
            except Exception as e:
                self.logger.error(f"Error shutting down provider '{name}'.", parameters={'error': str(e)}, exception=e)
        
        self.providers.clear()
        self.primary_provider = None
        self.fallback_providers.clear()
        self.logger.info("LLMManager shutdown complete.")

    def get_stats(self) -> Dict[str, Any]: # Added for consistency
        """Get LLMManager usage statistics."""
        return self._call_stats.copy()

    # For service container compatibility
    async def get_service_status(self) -> Dict[str, Any]:
        return await self.health_check()