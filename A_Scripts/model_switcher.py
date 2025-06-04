# legal_ai_system/core/model_switcher.py
#Dynamic Model Switching for Grok Models (and potentially others)

#This module provides runtime model switching capabilities, allowing users or the system
#to switch between configured LLM models based on task requirements or performance.


import time # Replaced logging with detailed_logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field # Added field
from datetime import datetime # For timestamps

# Use detailed_logging
from .detailed_logging import get_detailed_logger, LogCategory, detailed_log_function # Assuming in core
# Import from local llm_providers and config.grok_config
from .llm_providers import LLMManager, LLMConfig, LLMProviderEnum 
from ..config.grok_config import (
    GROK_MODELS_CONFIG, # Assuming this contains details for Grok models
    get_optimal_model_for_task as get_optimal_grok_model_for_task, # Alias for clarity
    is_reasoning_model as is_grok_reasoning_model, # Alias
    create_grok_config, # For creating specific Grok configs
    get_optimized_prompt as get_grok_optimized_prompt, # Alias
    get_available_grok_models
)

# Get a logger for this module
model_switcher_logger = get_detailed_logger("ModelSwitcher", LogCategory.LLM)


class TaskComplexity(Enum):
    """Task complexity levels for model selection"""
    SIMPLE = "simple"      # Quick tasks like citation formatting, simple Q&A
    MODERATE = "moderate"  # Standard analysis tasks, summarization
    COMPLEX = "complex"    # Multi-step reasoning, deep legal analysis

@dataclass
class ModelSwitchResult:
    """Result of a model switch operation"""
    success: bool
    previous_model: Optional[str] # Can be None if first switch
    new_model: str
    reason: str
    performance_impact_assessment: Optional[str] = None # Renamed for clarity

@dataclass
class ModelPerformanceRecord:
    """Record for model performance tracking."""
    calls: int = 0
    total_time_sec: float = 0.0
    successful_calls: int = 0
    
    @property
    def avg_time_sec(self) -> float:
        return self.total_time_sec / self.calls if self.calls > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return self.successful_calls / self.calls if self.calls > 0 else 0.0


class ModelSwitcher: # Renamed from GrokModelSwitcher for generality
    """
    Manages dynamic switching between LLM models based on task requirements or performance.
    This class can be adapted for various providers, but is initially focused on Grok models.
    """
    
    def __init__(self, llm_manager: LLMManager, default_provider_type: LLMProviderEnum = LLMProviderEnum.XAI):
        self.llm_manager = llm_manager
        self.default_provider_type = default_provider_type # e.g., XAI for Grok, OPENAI for OpenAI models
        
        # Attempt to get the API key from the primary provider's config if it's already set up
        self.api_key: Optional[str] = None
        if self.llm_manager.primary_provider and self.llm_manager.primary_provider.config:
            self.api_key = self.llm_manager.primary_provider.config.api_key
            self.current_model_name: Optional[str] = self.llm_manager.primary_provider.config.model
        else:
            self.current_model_name: Optional[str] = None # Will be set on first successful switch or init

        self.model_switch_history: List[Dict[str, Any]] = [] # Renamed for clarity
        self.task_to_model_cache: Dict[str, str] = {} # Renamed for clarity
        
        # Performance tracking per model
        self.model_performance_records: Dict[str, ModelPerformanceRecord] = defaultdict(ModelPerformanceRecord) # Renamed
        
        model_switcher_logger.info("ModelSwitcher initialized.", 
                                   parameters={'current_model': self.current_model_name, 
                                               'default_provider': self.default_provider_type.value})

    def _get_available_models_for_provider(self, provider_type: LLMProviderEnum) -> Dict[str, Dict[str, Any]]:
        """Helper to get available models for a specific provider type."""
        if provider_type == LLMProviderEnum.XAI:
            return get_available_grok_models()
        # Add logic for other providers if needed
        # elif provider_type == LLMProviderEnum.OPENAI:
        #     return OPENAI_MODELS_CONFIG 
        else:
            model_switcher_logger.warning(f"No specific model list for provider type {provider_type.value}. Relying on LLMManager's config.")
            # Fallback: if primary provider matches, return its model; otherwise, empty.
            if self.llm_manager.primary_provider and self.llm_manager.primary_provider.config.provider == provider_type:
                model_name = self.llm_manager.primary_provider.config.model
                return {model_name: {"model_name": model_name}} # Basic info
            return {}


    def suggest_model_for_task(self, task_type: str, complexity: TaskComplexity = TaskComplexity.MODERATE) -> str:
        """Suggest the optimal model for a specific task, considering provider type."""
        cache_key = f"{self.default_provider_type.value}_{task_type}_{complexity.value}"
        if cache_key in self.task_to_model_cache:
            return self.task_to_model_cache[cache_key]
        
        suggested_model_name = ""
        if self.default_provider_type == LLMProviderEnum.XAI:
            suggested_model_name = get_optimal_grok_model_for_task(task_type) # Uses Grok's internal logic
            # Adjust based on complexity for Grok models
            if complexity == TaskComplexity.COMPLEX and not is_grok_reasoning_model(suggested_model_name):
                if "grok-3-reasoning" in GROK_MODELS_CONFIG: # Check if reasoning model is defined
                    suggested_model_name = "grok-3-reasoning"
            elif complexity == TaskComplexity.SIMPLE and suggested_model_name == "grok-3-reasoning":
                 if "grok-3-mini" in GROK_MODELS_CONFIG:
                    suggested_model_name = "grok-3-mini"
        else:
            # Generic suggestion: use the current model or a default one
            # This part can be expanded with specific logic for other providers
            suggested_model_name = self.current_model_name or self.llm_manager.primary_config.model
            model_switcher_logger.warning(f"Using generic model suggestion for provider {self.default_provider_type.value}.",
                                       parameters={'suggested_model': suggested_model_name})

        self.task_to_model_cache[cache_key] = suggested_model_name
        model_switcher_logger.debug(f"Suggested model for task", 
                                   parameters={'task_type': task_type, 'complexity': complexity.value, 
                                               'provider': self.default_provider_type.value, 'model': suggested_model_name})
        return suggested_model_name

    @detailed_log_function(LogCategory.LLM)
    async def switch_to_model(self, model_name: str, reason: str = "Manual switch") -> ModelSwitchResult:
        """
        Switch the primary LLM provider's model configuration.
        Note: This requires LLMManager to support re-initializing or updating its primary provider.
        """
        previous_model = self.current_model_name
        
        available_models = self._get_available_models_for_provider(self.default_provider_type)
        if model_name not in available_models:
            msg = f"Model '{model_name}' is not available or not configured for provider '{self.default_provider_type.value}'."
            model_switcher_logger.error(msg)
            return ModelSwitchResult(success=False, previous_model=previous_model, new_model=model_name, reason=msg)

        if self.current_model_name == model_name:
            msg = f"Already using model '{model_name}'."
            model_switcher_logger.info(msg)
            return ModelSwitchResult(success=True, previous_model=previous_model, new_model=model_name, reason=msg)

        if not self.api_key and self.default_provider_type != LLMProviderEnum.OLLAMA : # Ollama might not need API key
            msg = f"API key not available for provider {self.default_provider_type.value}. Cannot switch model."
            model_switcher_logger.error(msg)
            return ModelSwitchResult(success=False, previous_model=previous_model, new_model=model_name, reason=msg)

        try:
            new_llm_config: Optional[LLMConfig] = None
            if self.default_provider_type == LLMProviderEnum.XAI:
                new_llm_config = create_grok_config(model_name=model_name, api_key=self.api_key)
            # Add elif for other providers like OpenAI if they have specific config creation
            else: # Generic attempt for other providers, assuming model name is the main change
                if self.llm_manager.primary_provider:
                    current_primary_config = self.llm_manager.primary_provider.config
                    new_llm_config = current_primary_config.copy(update={'model': model_name})
                else: # Cannot create a new config if no primary provider to base it on
                     msg = "Cannot switch model: No primary LLM provider configured in LLMManager."
                     model_switcher_logger.error(msg)
                     return ModelSwitchResult(success=False, previous_model=previous_model, new_model=model_name, reason=msg)


            # This is the critical part: LLMManager needs to handle this.
            # For now, we assume LLMManager can re-initialize its primary provider or has a method for this.
            if hasattr(self.llm_manager, 'update_primary_provider_config'):
                await self.llm_manager.update_primary_provider_config(new_llm_config) # Ideal scenario
            elif hasattr(self.llm_manager, 'initialize_provider'): # Alternative: re-init specific provider
                self.llm_manager.primary_config = new_llm_config
                self.llm_manager.primary_provider = self.llm_manager._create_provider(new_llm_config)
                await self.llm_manager.primary_provider.initialize()
                self.llm_manager.providers[new_llm_config.provider.value] = self.llm_manager.primary_provider
            else:
                # Fallback: Log the intent and update local state. The user/system needs to ensure LLMManager is reconfigured.
                model_switcher_logger.warning("LLMManager does not have a direct method to update primary provider config. "
                                           "Switcher will track the change, but LLMManager might need manual reconfiguration or restart.",
                                           parameters={'new_model': model_name})
                # Update LLMManager's primary_config directly (less ideal but might work if LLMManager re-reads it)
                self.llm_manager.primary_config = new_llm_config


            self.current_model_name = model_name
            self.model_switch_history.append({
                "timestamp": datetime.now().isoformat(),
                "from_model": previous_model,
                "to_model": model_name,
                "reason": reason
            })
            
            performance_impact = self._assess_performance_impact(previous_model, model_name)
            model_switcher_logger.info(f"Successfully switched model.", 
                                      parameters={'from': previous_model, 'to': model_name, 'reason': reason, 'impact': performance_impact})
            return ModelSwitchResult(success=True, previous_model=previous_model, new_model=model_name, 
                                     reason=reason, performance_impact_assessment=performance_impact)
            
        except Exception as e:
            model_switcher_logger.error(f"Failed to switch model to '{model_name}'.", parameters={'error': str(e)}, exception=e)
            return ModelSwitchResult(success=False, previous_model=previous_model, new_model=model_name, reason=f"Switch failed: {str(e)}")

    @detailed_log_function(LogCategory.LLM)
    async def switch_for_task(self, task_type: str, complexity: TaskComplexity = TaskComplexity.MODERATE) -> ModelSwitchResult:
        """Automatically switch to the optimal model for a given task."""
        optimal_model = self.suggest_model_for_task(task_type, complexity)
        reason = f"Optimizing for task: '{task_type}' (Complexity: {complexity.value})"
        return await self.switch_to_model(optimal_model, reason)
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the currently active model."""
        if not self.current_model_name:
            return {"error": "No model currently active or configured in switcher."}

        available_models = self._get_available_models_for_provider(self.default_provider_type)
        model_config = available_models.get(self.current_model_name)

        if not model_config:
            return {"error": f"Current model '{self.current_model_name}' not found in configurations for provider '{self.default_provider_type.value}'."}
        
        performance_record = self.model_performance_records[self.current_model_name]
        
        return {
            "model_name": self.current_model_name,
            "provider": self.default_provider_type.value,
            "use_case": model_config.get("use_case", "N/A"),
            "reasoning_capable": model_config.get("reasoning", False),
            "context_length": model_config.get("context_length", "N/A"),
            "max_tokens": model_config.get("max_tokens", "N/A"),
            "performance": {
                "calls": performance_record.calls,
                "avg_time_sec": round(performance_record.avg_time_sec, 3),
                "success_rate": round(performance_record.success_rate, 3)
            }
        }

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available models for the default provider."""
        models_info = {}
        available_models = self._get_available_models_for_provider(self.default_provider_type)

        for model_name, config_details in available_models.items():
            performance_record = self.model_performance_records[model_name]
            models_info[model_name] = {
                **config_details,
                "performance": {
                    "calls": performance_record.calls,
                    "avg_time_sec": round(performance_record.avg_time_sec, 3),
                    "success_rate": round(performance_record.success_rate, 3)
                },
                "is_current": model_name == self.current_model_name
            }
        return models_info
    
    def get_optimized_prompt_for_current_model(self, prompt_type: str, **kwargs) -> str:
        """Get prompt optimized for the current model, specific to provider type."""
        if not self.current_model_name:
            raise ValueError("No current model set. Cannot get optimized prompt.")

        if self.default_provider_type == LLMProviderEnum.XAI:
            return get_grok_optimized_prompt(prompt_type, self.current_model_name, **kwargs)
        # Add logic for other providers if they have specific prompt optimization
        else:
            # Generic fallback: could be a simple format or raise error
            model_switcher_logger.warning(f"No specific prompt optimization for provider {self.default_provider_type.value}. Using generic prompt structure.")
            # This needs a generic prompt store or logic if we expand beyond Grok.
            # For now, let's assume a simple placeholder if not Grok.
            return f"Task: {prompt_type}\nContext: {kwargs.get('document_text', '')[:200]}..."


    def _assess_performance_impact(self, from_model_name: Optional[str], to_model_name: str) -> str:
        """Assess the performance impact of switching models."""
        if not from_model_name or not to_model_name:
            return "Performance impact unknown due to missing model information."

        available_models = self._get_available_models_for_provider(self.default_provider_type)
        from_config = available_models.get(from_model_name)
        to_config = available_models.get(to_model_name)

        if not from_config or not to_config:
            return "Performance impact unknown due to incomplete model configuration."

        from_reasoning = from_config.get("reasoning", False)
        to_reasoning = to_config.get("reasoning", False)
        
        # Simplistic assessment based on reasoning capability (often correlates with speed/cost)
        if not from_reasoning and to_reasoning:
            return "Potential for slower but more thorough/accurate analysis."
        elif from_reasoning and not to_reasoning:
            return "Potential for faster but possibly less detailed/accurate analysis."
        elif from_reasoning == to_reasoning:
            return "Performance characteristics expected to be broadly similar; check specific model benchmarks."
        return "Performance impact assessment inconclusive."

    @detailed_log_function(LogCategory.LLM)
    def record_task_performance(self, model_name: str, task_duration_sec: float, success: bool): # Renamed parameter
        """Record performance metrics for a task completed by a specific model."""
        if model_name not in self.model_performance_records:
            # This can happen if a model is used without being formally "switched to" via this switcher
            # or if it's a new model. Initialize its record.
            self.model_performance_records[model_name] = ModelPerformanceRecord()
            
        perf_record = self.model_performance_records[model_name]
        perf_record.calls += 1
        perf_record.total_time_sec += task_duration_sec
        if success:
            perf_record.successful_calls += 1
        
        model_switcher_logger.debug("Task performance recorded", 
                                   parameters={'model': model_name, 'duration_sec': task_duration_sec, 'success': success,
                                               'new_avg_time': perf_record.avg_time_sec, 'new_success_rate': perf_record.success_rate})

    def get_model_recommendations(self) -> List[Dict[str, Any]]:
        """Get model recommendations based on performance history."""
        recommendations = []
        available_models = self._get_available_models_for_provider(self.default_provider_type)

        for model_name, perf_record in self.model_performance_records.items():
            if perf_record.calls > 0: # Only recommend models with performance data
                model_config_details = available_models.get(model_name, {})
                
                recommendation_text = self._generate_recommendation_text(model_name, perf_record)
                
                recommendations.append({
                    "model_name": model_name,
                    "provider": self.default_provider_type.value,
                    "use_case": model_config_details.get("use_case", "N/A"),
                    "reasoning_capable": model_config_details.get("reasoning", False),
                    "avg_time_sec": round(perf_record.avg_time_sec, 3),
                    "success_rate": round(perf_record.success_rate, 3),
                    "total_calls": perf_record.calls,
                    "recommendation_text": recommendation_text
                })
        
        # Sort by a composite score: higher success rate is better, lower avg time is better.
        recommendations.sort(key=lambda x: (x["success_rate"], -x["avg_time_sec"]), reverse=True)
        return recommendations

    def _generate_recommendation_text(self, model_name: str, perf_record: ModelPerformanceRecord) -> str:
        """Generate recommendation text for a model based on its performance."""
        if perf_record.success_rate >= 0.9 and perf_record.avg_time_sec < 5.0 and perf_record.calls >= 10:
            return "Excellent performance & reliability. Highly recommended for similar tasks."
        elif perf_record.success_rate >= 0.8 and perf_record.calls >= 5:
            return "Good performance. Suitable for regular use."
        elif perf_record.success_rate >= 0.6:
            return "Moderate performance. Consider for specific, non-critical use cases or with supervision."
        elif perf_record.calls < 5:
            return "Limited performance data. More usage needed for a robust recommendation."
        else:
            return "Performance may be suboptimal. Use with caution or consider alternatives."

# Convenience functions for common switching patterns (can be adapted for generic ModelSwitcher)
async def switch_to_reasoning_mode(switcher: ModelSwitcher) -> ModelSwitchResult:
    """Switch to a reasoning-capable model for complex analysis."""
    # This needs to find a reasoning model for the switcher's default_provider_type
    available_models = switcher._get_available_models_for_provider(switcher.default_provider_type)
    reasoning_model = next((name for name, conf in available_models.items() if conf.get("reasoning")), None)
    
    if reasoning_model:
        return await switcher.switch_to_model(reasoning_model, "Switching to reasoning mode for complex analysis.")
    else:
        model_switcher_logger.warning(f"No reasoning model found for provider {switcher.default_provider_type.value}. Cannot switch to reasoning mode.")
        return ModelSwitchResult(success=False, previous_model=switcher.current_model_name, new_model=switcher.current_model_name or "", 
                                 reason=f"No reasoning model available for {switcher.default_provider_type.value}")

async def switch_to_fast_mode(switcher: ModelSwitcher) -> ModelSwitchResult:
    """Switch to a fast model for quick tasks."""
    # This needs to find a non-reasoning (typically faster) model
    available_models = switcher._get_available_models_for_provider(switcher.default_provider_type)
    # Prioritize 'mini' models or non-reasoning ones. This is heuristic.
    fast_model = next((name for name, conf in available_models.items() if "mini" in name.lower() and not conf.get("reasoning")), None)
    if not fast_model:
        fast_model = next((name for name, conf in available_models.items() if not conf.get("reasoning")), None)
    
    if fast_model:
        return await switcher.switch_to_model(fast_model, "Switching to fast mode for quick tasks.")
    else: # Fallback to any available model if no clear "fast" model
        default_model = list(available_models.keys())[0] if available_models else None
        if default_model:
            return await switcher.switch_to_model(default_model, "Switching to default model (no specific fast model found).")
        model_switcher_logger.warning(f"No fast model found for provider {switcher.default_provider_type.value}. Cannot switch to fast mode.")
        return ModelSwitchResult(success=False, previous_model=switcher.current_model_name, new_model=switcher.current_model_name or "", 
                                 reason=f"No fast model available for {switcher.default_provider_type.value}")


async def auto_switch_for_legal_analysis(switcher: ModelSwitcher, document_length: int) -> ModelSwitchResult:
    """Automatically switch model based on document complexity for legal analysis task."""
    complexity: TaskComplexity
    if document_length > 10000:  # Example threshold for very long/complex documents
        complexity = TaskComplexity.COMPLEX
    elif document_length > 3000:
        complexity = TaskComplexity.MODERATE  
    else:
        complexity = TaskComplexity.SIMPLE
    
    return await switcher.switch_for_task("legal_analysis", complexity)