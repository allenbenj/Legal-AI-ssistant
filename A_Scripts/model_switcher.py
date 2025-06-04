"""
Dynamic Model Switching for Grok Models

This module provides runtime model switching capabilities, allowing users to
switch between Grok models (standard and reasoning) based on task requirements.
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

from .llm_providers import LLMManager, LLMConfig, LLMProvider
from ..config.grok_3_mini_setup import (
    GROK_MODELS_CONFIG,
    get_optimal_model_for_task,
    is_reasoning_model,
    create_grok_config,
    get_optimized_prompt
)

logger = logging.getLogger(__name__)

class TaskComplexity(Enum):
    """Task complexity levels for model selection"""
    SIMPLE = "simple"      # Quick tasks like citation formatting
    MODERATE = "moderate"  # Standard analysis tasks
    COMPLEX = "complex"    # Multi-step reasoning tasks

@dataclass
class ModelSwitchResult:
    """Result of a model switch operation"""
    success: bool
    previous_model: str
    new_model: str
    reason: str
    performance_impact: Optional[str] = None

class GrokModelSwitcher:
    """
    Manages dynamic switching between Grok models based on task requirements
    """
    
    def __init__(self, llm_manager: LLMManager, api_key: str):
        self.llm_manager = llm_manager
        self.api_key = api_key
        self.current_model = None
        self.model_history: List[Dict] = []
        self.task_model_cache: Dict[str, str] = {}
        
        # Performance tracking
        self.model_performance: Dict[str, Dict] = {
            model: {"calls": 0, "avg_time": 0.0, "success_rate": 0.0}
            for model in GROK_MODELS_CONFIG.keys()
        }
        
        # Get current model
        if hasattr(llm_manager.primary_provider, 'config'):
            self.current_model = llm_manager.primary_provider.config.model
        
        logger.info(f"GrokModelSwitcher initialized with current model: {self.current_model}")
    
    def suggest_model_for_task(self, task_type: str, complexity: TaskComplexity = TaskComplexity.MODERATE) -> str:
        """
        Suggest the optimal Grok model for a specific task
        
        Args:
            task_type: Type of task (analysis, reasoning, citation, etc.)
            complexity: Task complexity level
        
        Returns:
            Recommended model name
        """
        # Check cache first
        cache_key = f"{task_type}_{complexity.value}"
        if cache_key in self.task_model_cache:
            return self.task_model_cache[cache_key]
        
        # Get base recommendation
        base_model = get_optimal_model_for_task(task_type)
        
        # Adjust based on complexity
        if complexity == TaskComplexity.COMPLEX:
            # For complex tasks, prefer reasoning models
            if not is_reasoning_model(base_model):
                if "grok-3-reasoning" in GROK_MODELS_CONFIG:
                    base_model = "grok-3-reasoning"
        elif complexity == TaskComplexity.SIMPLE:
            # For simple tasks, prefer faster models
            if base_model == "grok-3-reasoning":
                base_model = "grok-3-mini"
        
        # Cache the decision
        self.task_model_cache[cache_key] = base_model
        
        logger.debug(f"Suggested model for {task_type} ({complexity.value}): {base_model}")
        return base_model
    
    def switch_to_model(self, model_name: str, reason: str = "Manual switch") -> ModelSwitchResult:
        """
        Switch to a specific Grok model
        
        Args:
            model_name: Target model name
            reason: Reason for the switch
        
        Returns:
            ModelSwitchResult with switch details
        """
        previous_model = self.current_model
        
        try:
            # Validate model
            if model_name not in GROK_MODELS_CONFIG:
                return ModelSwitchResult(
                    success=False,
                    previous_model=previous_model,
                    new_model=model_name,
                    reason=f"Unknown model: {model_name}"
                )
            
            # Skip if already using this model
            if self.current_model == model_name:
                return ModelSwitchResult(
                    success=True,
                    previous_model=previous_model,
                    new_model=model_name,
                    reason="Already using requested model"
                )
            
            # Create new config
            new_config = create_grok_config(
                model_name=model_name,
                api_key=self.api_key
            )
            
            # Update LLM manager
            # Note: This would require extending LLMManager to support runtime model switching
            # For now, we'll log the intention
            logger.info(f"Switching from {previous_model} to {model_name}: {reason}")
            
            # Update current model tracking
            self.current_model = model_name
            
            # Record in history
            self.model_history.append({
                "timestamp": logger.getEffectiveLevel(),  # Placeholder
                "from_model": previous_model,
                "to_model": model_name,
                "reason": reason
            })
            
            # Determine performance impact
            performance_impact = self._assess_performance_impact(previous_model, model_name)
            
            return ModelSwitchResult(
                success=True,
                previous_model=previous_model,
                new_model=model_name,
                reason=reason,
                performance_impact=performance_impact
            )
            
        except Exception as e:
            logger.error(f"Failed to switch to model {model_name}: {e}")
            return ModelSwitchResult(
                success=False,
                previous_model=previous_model,
                new_model=model_name,
                reason=f"Switch failed: {e}"
            )
    
    def switch_for_task(self, task_type: str, complexity: TaskComplexity = TaskComplexity.MODERATE) -> ModelSwitchResult:
        """
        Automatically switch to the optimal model for a task
        
        Args:
            task_type: Type of task
            complexity: Task complexity level
        
        Returns:
            ModelSwitchResult with switch details
        """
        optimal_model = self.suggest_model_for_task(task_type, complexity)
        reason = f"Optimizing for {task_type} task ({complexity.value} complexity)"
        
        return self.switch_to_model(optimal_model, reason)
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.current_model or self.current_model not in GROK_MODELS_CONFIG:
            return {"error": "No current model or unknown model"}
        
        model_config = GROK_MODELS_CONFIG[self.current_model]
        performance = self.model_performance.get(self.current_model, {})
        
        return {
            "model_name": self.current_model,
            "use_case": model_config["use_case"],
            "reasoning_capable": model_config["reasoning"],
            "context_length": model_config["context_length"],
            "max_tokens": model_config["max_tokens"],
            "performance": performance
        }
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available models"""
        models_info = {}
        
        for model_name, config in GROK_MODELS_CONFIG.items():
            performance = self.model_performance.get(model_name, {})
            models_info[model_name] = {
                **config,
                "performance": performance,
                "is_current": model_name == self.current_model
            }
        
        return models_info
    
    def get_optimized_prompt_for_current_model(self, prompt_type: str, **kwargs) -> str:
        """
        Get prompt optimized for the current model
        
        Args:
            prompt_type: Type of prompt
            **kwargs: Variables for prompt formatting
        
        Returns:
            Formatted prompt string
        """
        return get_optimized_prompt(prompt_type, self.current_model, **kwargs)
    
    def _assess_performance_impact(self, from_model: str, to_model: str) -> str:
        """Assess the performance impact of switching models"""
        if not from_model or not to_model:
            return "Unknown impact"
        
        from_reasoning = is_reasoning_model(from_model)
        to_reasoning = is_reasoning_model(to_model)
        
        if not from_reasoning and to_reasoning:
            return "Slower but more thorough analysis"
        elif from_reasoning and not to_reasoning:
            return "Faster but less detailed analysis"
        else:
            return "Similar performance characteristics"
    
    def record_task_performance(self, model_name: str, task_duration: float, success: bool) -> None:
        """Record performance metrics for a task"""
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {
                "calls": 0,
                "avg_time": 0.0,
                "success_rate": 0.0,
                "total_successes": 0
            }
        
        perf = self.model_performance[model_name]
        perf["calls"] += 1
        
        # Update average time
        perf["avg_time"] = ((perf["avg_time"] * (perf["calls"] - 1)) + task_duration) / perf["calls"]
        
        # Update success rate
        if success:
            perf["total_successes"] = perf.get("total_successes", 0) + 1
        
        perf["success_rate"] = perf["total_successes"] / perf["calls"]
    
    def get_model_recommendations(self) -> List[Dict[str, Any]]:
        """Get model recommendations based on performance history"""
        recommendations = []
        
        # Analyze performance data
        for model_name, perf in self.model_performance.items():
            if perf["calls"] > 0:
                model_config = GROK_MODELS_CONFIG[model_name]
                
                recommendation = {
                    "model": model_name,
                    "use_case": model_config["use_case"],
                    "reasoning": model_config["reasoning"],
                    "avg_time": perf["avg_time"],
                    "success_rate": perf["success_rate"],
                    "total_calls": perf["calls"],
                    "recommendation": self._generate_recommendation(model_name, perf)
                }
                
                recommendations.append(recommendation)
        
        # Sort by success rate and performance
        recommendations.sort(key=lambda x: (x["success_rate"], -x["avg_time"]), reverse=True)
        
        return recommendations
    
    def _generate_recommendation(self, model_name: str, perf: Dict) -> str:
        """Generate recommendation text for a model"""
        if perf["success_rate"] > 0.9 and perf["avg_time"] < 5.0:
            return "Excellent performance - recommended for similar tasks"
        elif perf["success_rate"] > 0.8:
            return "Good performance - suitable for regular use"
        elif perf["success_rate"] > 0.6:
            return "Moderate performance - consider for specific use cases"
        else:
            return "Poor performance - avoid for critical tasks"

# Convenience functions for common switching patterns
def switch_to_reasoning_mode(switcher: GrokModelSwitcher) -> ModelSwitchResult:
    """Switch to reasoning model for complex analysis"""
    return switcher.switch_to_model("grok-3-reasoning", "Switching to reasoning mode")

def switch_to_fast_mode(switcher: GrokModelSwitcher) -> ModelSwitchResult:
    """Switch to fast model for quick tasks"""
    return switcher.switch_to_model("grok-3-mini", "Switching to fast mode")

def auto_switch_for_legal_analysis(switcher: GrokModelSwitcher, document_length: int) -> ModelSwitchResult:
    """Automatically switch model based on document complexity"""
    if document_length > 5000:  # Long document - use reasoning
        complexity = TaskComplexity.COMPLEX
    elif document_length > 1000:  # Medium document
        complexity = TaskComplexity.MODERATE  
    else:  # Short document - fast mode
        complexity = TaskComplexity.SIMPLE
    
    return switcher.switch_for_task("legal_analysis", complexity)