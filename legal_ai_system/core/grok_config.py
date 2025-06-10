# legal_ai_system/config/grok_config.py
# Grok LLM Specific Configuration and Setup

# This module provides optimized configuration for using xAI's Grok models
# as LLM providers for the Legal AI System.


from typing import Any, Dict, List, Optional, Union

from .llm_providers import LLMConfig, LLMProviderEnum

# Simplified imports from local modules. These modules are part of the
# ``legal_ai_system`` package, so standard relative imports are sufficient.
from .settings import LegalAISettings

# Grok model configurations
GROK_MODELS_CONFIG: Dict[str, Dict[str, Any]] = {
    "grok-3-mini": {
        "model_name": "grok-3-mini",
        "context_length": 8192,
        "max_tokens": 4096,
        "temperature": 0.7,
        "use_case": "Fast, efficient legal analysis",
        "reasoning": False,
    },
    "grok-3-reasoning": {
        "model_name": "grok-3-reasoning",
        "context_length": 8192,
        "max_tokens": 4096,
        "temperature": 0.7,  # Potentially lower for reasoning tasks
        "use_case": "Complex legal reasoning and analysis",
        "reasoning": True,
    },
    "grok-2-1212": {  # Assuming this is another model, ensure its name is accurate
        "model_name": "grok-2-1212",  # Placeholder name, update if different
        "context_length": 8192,
        "max_tokens": 4096,
        "temperature": 0.7,
        "use_case": "Balanced performance and reasoning",
        "reasoning": False,  # Or True, depending on the model's capability
    },
}

# Default configuration (can be switched at runtime)
DEFAULT_GROK_MODEL: str = "grok-3-mini"

# Optimized parameters for legal text (applies to all models)
LEGAL_ANALYSIS_PARAMS: Dict[str, float] = {
    "legal_temperature": 0.3,  # Lower temperature for legal analysis
    "citation_temperature": 0.1,  # Very low for citation formatting
    "summary_temperature": 0.5,  # Moderate for summaries
    "reasoning_temperature": 0.2,  # Low for step-by-step reasoning
}

# Rate limiting (adjust based on your xAI plan)
RATE_LIMITS: Dict[str, Union[int, float]] = {  # type: ignore[valid-type]
    "requests_per_minute": 60,
    "tokens_per_minute": 100000,
    "retry_attempts": 3,
    "retry_delay": 1.0,  # seconds
    "timeout": 60,  # seconds
}


def create_grok_config(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = "https://api.x.ai/v1",
) -> LLMConfig:
    """
    Create optimized LLMConfig for any Grok model.

    Args:
        model_name: Grok model name (e.g., "grok-3-mini"). Defaults to DEFAULT_GROK_MODEL.
        api_key: Your xAI API key.
        base_url: Optional custom xAI API endpoint.

    Returns:
        LLMConfig optimized for the specified Grok model.
    """
    resolved_model_name = model_name or DEFAULT_GROK_MODEL

    if resolved_model_name not in GROK_MODELS_CONFIG:
        raise ValueError(
            f"Unknown Grok model: {resolved_model_name}. Available: {list(GROK_MODELS_CONFIG.keys())}"
        )

    model_specific_config = GROK_MODELS_CONFIG[resolved_model_name]

    return LLMConfig(
        provider=LLMProviderEnum.XAI,  # XAI provider enum
        model=model_specific_config["model_name"],
        api_key=api_key,
        base_url=base_url,
        temperature=model_specific_config["temperature"],
        max_tokens=model_specific_config["max_tokens"],
        timeout=int(RATE_LIMITS["timeout"]),  # Ensure int
        retry_attempts=int(RATE_LIMITS["retry_attempts"]),  # Ensure int
        retry_delay=float(RATE_LIMITS["retry_delay"]),  # Ensure float
    )


# Backward compatibility


def create_grok_3_mini_config(
    api_key: str, base_url: str = "https://api.x.ai/v1"
) -> LLMConfig:
    """Backward compatibility function for Grok-3-Mini."""
    return create_grok_config("grok-3-mini", api_key, base_url)


# Renamed from get_grok_3_mini_settings for generality
def get_grok_primary_settings() -> Dict[str, Any]:
    """
    Get environment settings optimized for a Grok model as primary provider.
    Uses DEFAULT_GROK_MODEL.

    Returns:
        Dictionary of settings for LegalAISettings.
    """
    if DEFAULT_GROK_MODEL not in GROK_MODELS_CONFIG:
        raise ValueError(
            f"Default Grok model '{DEFAULT_GROK_MODEL}' not found in configurations."
        )

    default_model_config = GROK_MODELS_CONFIG[DEFAULT_GROK_MODEL]

    return {
        # Primary LLM Provider
        "llm_provider": "xai",  # Hardcoded as this is grok_config
        "llm_model": default_model_config["model_name"],
        "llm_temperature": default_model_config["temperature"],
        "llm_max_tokens": default_model_config["max_tokens"],
        # xAI specific settings
        # Redundant with llm_model if provider is xai
        "xai_model": default_model_config["model_name"],
        "xai_base_url": "https://api.x.ai/v1",
        # Fallback to Ollama for privacy-sensitive operations or if xAI fails
        "fallback_provider": "ollama",
        "fallback_model": "llama3.2",  # Example fallback
        # Optimized context management for the default Grok model
        # Leave buffer (e.g. 75%)
        "max_context_tokens": int(default_model_config["context_length"] * 0.75),
        # Summarize if >50% context
        "auto_summarize_threshold": int(default_model_config["context_length"] * 0.5),
        # Performance optimizations (consider rate limits)
        "max_concurrent_documents": 2,  # Conservative for rate limits
        "batch_size": 5,  # Depends on task, may not apply to all LLM calls
    }


# Allow specifying model
def create_grok_legal_settings(
    api_key: str, model_name: Optional[str] = None
) -> LegalAISettings:
    """
    Create complete LegalAISettings optimized for a specific Grok model.

    Args:
        api_key: Your xAI API key.
        model_name: Specific Grok model to configure for. Defaults to DEFAULT_GROK_MODEL.

    Returns:
        LegalAISettings instance configured for the Grok model.
    """
    target_model = model_name or DEFAULT_GROK_MODEL
    if target_model not in GROK_MODELS_CONFIG:
        raise ValueError(
            f"Target Grok model '{target_model}' not found in configurations."
        )

    model_specific_config = GROK_MODELS_CONFIG[target_model]

    # Start with general Grok settings
    settings_dict = get_grok_primary_settings()

    # Override with specific model settings
    settings_dict["llm_model"] = model_specific_config["model_name"]
    settings_dict["llm_temperature"] = model_specific_config["temperature"]
    settings_dict["llm_max_tokens"] = model_specific_config["max_tokens"]
    settings_dict["xai_model"] = model_specific_config["model_name"]
    settings_dict["max_context_tokens"] = int(
        model_specific_config["context_length"] * 0.75
    )
    settings_dict["auto_summarize_threshold"] = int(
        model_specific_config["context_length"] * 0.5
    )

    # Add API key
    settings_dict["xai_api_key"] = api_key

    return LegalAISettings(**settings_dict)


# Model switching functionality


# Renamed parameter
def switch_grok_model_config(
    current_config: LLMConfig, new_model_name: str
) -> LLMConfig:
    """
    Switch to a different Grok model while preserving other settings like API key and base URL.

    Args:
        current_config: Current LLMConfig (must be for XAI provider).
        new_model_name: New Grok model name.

    Returns:
        New LLMConfig with updated model.
    """
    if current_config.provider != LLMProviderEnum.XAI:
        raise ValueError(
            "Current config must be for XAI provider to switch Grok models."
        )

    return create_grok_config(
        model_name=new_model_name,
        api_key=current_config.api_key,
        base_url=current_config.base_url,
    )


def get_available_grok_models() -> Dict[str, Dict[str, Any]]:
    """Get list of available Grok models with their capabilities."""
    return GROK_MODELS_CONFIG.copy()


def is_reasoning_model(model_name: str) -> bool:
    """Check if a model supports reasoning capabilities."""
    if model_name in GROK_MODELS_CONFIG:
        # Use .get for safety
        return GROK_MODELS_CONFIG[model_name].get("reasoning", False)
    return False


def get_optimal_model_for_task(task_type: str) -> str:
    """Return the recommended Grok model name for a given task type."""
    task_recommendations = {
        "reasoning": "grok-3-reasoning",
        "constitutional_analysis": "grok-3-reasoning",
        "case_analysis": "grok-3-reasoning",
        "violation_detection": "grok-3-reasoning",
        "citation": "grok-3-mini",
        "summary": "grok-3-mini",
        "extraction": "grok-3-mini",
        "classification": "grok-3-mini",
        "general": "grok-3-mini",
    }

    return task_recommendations.get(task_type, DEFAULT_GROK_MODEL)


# Grok prompt templates optimized for different models and legal analysis
GROK_PROMPTS: Dict[str, str] = {
    "legal_analysis": """
    You are a legal AI assistant. Analyze the following legal document with precision and accuracy.
    
    Focus on:
    1. Legal violations and misconduct
    2. Procedural issues
    3. Constitutional concerns
    4. Factual inconsistencies
    
    Document to analyze:
    {document_text}
    
    Provide a structured analysis with citations and reasoning.
    """,
    "legal_analysis_reasoning": """
    You are a legal AI assistant with advanced reasoning capabilities. Analyze the following legal document step-by-step.
    
    Use the following reasoning framework:
    1. **Initial Assessment**: What type of legal document is this?
    2. **Key Issues Identification**: What are the main legal issues present?
    3. **Constitutional Analysis**: Are there any constitutional concerns?
    4. **Procedural Review**: Were proper procedures followed?
    5. **Evidence Evaluation**: Is the evidence properly handled and presented?
    6. **Legal Standards Application**: What legal standards apply and are they met?
    7. **Conclusion**: What violations or issues can be conclusively identified?
    
    Document to analyze:
    {document_text}
    
    Think through each step carefully and provide detailed reasoning for your conclusions.
    """,
    "violation_detection": """
    As a legal expert, examine this document for potential legal violations.
    
    Look for:
    - Constitutional violations
    - Procedural violations
    - Ethical violations
    - Evidence handling issues
    
    Document:
    {document_text}
    
    List each violation with:
    1. Type of violation
    2. Specific evidence
    3. Relevant legal standards
    4. Severity assessment
    """,
    "violation_detection_reasoning": """
    As a legal expert with step-by-step reasoning capabilities, systematically examine this document for potential legal violations.
    
    Follow this analytical process:
    1. **Document Classification**: What type of legal document/proceeding is this?
    2. **Applicable Legal Framework**: What laws, rules, and standards apply?
    3. **Constitutional Review**: Examine for constitutional violations (4th, 5th, 6th, 14th Amendment issues)
    4. **Procedural Analysis**: Were proper legal procedures followed?
    5. **Ethical Standards Review**: Any professional conduct violations?
    6. **Evidence Chain Analysis**: Proper evidence handling and chain of custody?
    7. **Due Process Evaluation**: Were due process rights respected?
    8. **Final Assessment**: Rank violations by severity and likelihood
    
    Document:
    {document_text}
    
    Work through each step methodically, explaining your reasoning for each potential violation identified.
    """,
    "citation_formatting": """
    Format the following legal citations according to Bluebook style.
    Be extremely precise with formatting, punctuation, and abbreviations.
    
    Citations to format:
    {citations}
    
    Return only the properly formatted citations.
    """,
    "document_summary": """
    Summarize this legal document concisely but comprehensively.
    Include key parties, issues, holdings, and implications.
    
    Document:
    {document_text}
    
    Provide a structured summary with:
    - Case/Document overview
    - Key legal issues
    - Main arguments
    - Conclusions/Holdings
    - Implications
    """,
}


def get_optimized_prompt(
    prompt_type: str, model_name: Optional[str] = None, **kwargs
) -> str:
    """
    Get optimized prompt for specific task and model.

    Args:
        prompt_type: Type of prompt (e.g., "legal_analysis", "violation_detection").
        model_name: Grok model name (auto-detects reasoning capability). If None, uses default model.
        **kwargs: Variables to substitute in the prompt.

    Returns:
        Formatted prompt string optimized for the model.
    """
    target_model_name = model_name or DEFAULT_GROK_MODEL

    # Determine if we should use reasoning variant
    use_reasoning = is_reasoning_model(target_model_name)

    # Try reasoning version first if available and appropriate
    reasoning_prompt_key = f"{prompt_type}_reasoning"
    if use_reasoning and reasoning_prompt_key in GROK_PROMPTS:
        return GROK_PROMPTS[reasoning_prompt_key].format(**kwargs)

    # Fall back to standard prompt
    if prompt_type not in GROK_PROMPTS:
        raise ValueError(
            f"Unknown prompt type: {prompt_type}. Available: {list(GROK_PROMPTS.keys())}"
        )

    return GROK_PROMPTS[prompt_type].format(**kwargs)


def get_available_prompt_types() -> List[str]:
    """Get list of available base prompt types (without _reasoning suffix)."""
    base_types = set()
    for prompt_key in GROK_PROMPTS.keys():
        if prompt_key.endswith("_reasoning"):
            base_types.add(prompt_key[: -len("_reasoning")])
        else:
            base_types.add(prompt_key)
    return sorted(list(base_types))


# Usage example for setting up Grok-3-Mini configuration as primary


def setup_default_grok_system(api_key: str) -> Dict[str, Any]:
    """
    Complete setup function for a Grok-based Legal AI System using the default Grok model.

    Args:
        api_key: Your xAI API key.

    Returns:
        Dictionary with LLMConfig, LegalAISettings, available prompts, and notes.
    """
    llm_config = create_grok_config(api_key=api_key)  # Uses DEFAULT_GROK_MODEL
    settings = create_grok_legal_settings(api_key=api_key)  # Uses DEFAULT_GROK_MODEL

    if DEFAULT_GROK_MODEL not in GROK_MODELS_CONFIG:
        raise ValueError(
            f"Default Grok model '{DEFAULT_GROK_MODEL}' is not defined in GROK_MODELS_CONFIG."
        )
    default_model_details = GROK_MODELS_CONFIG[DEFAULT_GROK_MODEL]

    return {
        "llm_config": llm_config,
        "settings": settings,
        "prompts": GROK_PROMPTS,  # Provide all prompts
        "optimization_notes": [
            f"Grok system optimized for {default_model_details['model_name']}.",
            f"Context window: {default_model_details['context_length']} tokens.",
            f"Primary use case: {default_model_details['use_case']}.",
            "Lower temperature settings generally recommended for legal precision.",
            "Ollama fallback configured for sensitive documents or API issues.",
        ],
    }


# Environment variables template for Grok-3-Mini setup
GROK_ENV_TEMPLATE = f"""
# Grok Optimized Configuration ({DEFAULT_GROK_MODEL} as default)
LLM_PROVIDER=xai
LLM_MODEL={DEFAULT_GROK_MODEL}
LLM_TEMPERATURE={GROK_MODELS_CONFIG.get(DEFAULT_GROK_MODEL, {}).get("temperature", 0.7)}
LLM_MAX_TOKENS={GROK_MODELS_CONFIG.get(DEFAULT_GROK_MODEL, {}).get("max_tokens", 4096)}

# xAI Settings
XAI_API_KEY="your_xai_api_key_here" # Replace with your actual key
XAI_BASE_URL=https://api.x.ai/v1
XAI_MODEL={DEFAULT_GROK_MODEL} # Can be overridden by LLM_MODEL if provider is xai

# Fallback to Ollama for privacy
FALLBACK_PROVIDER=ollama
FALLBACK_MODEL=llama3.2

# Optimized context management
MAX_CONTEXT_TOKENS={int(GROK_MODELS_CONFIG.get(DEFAULT_GROK_MODEL, {}).get("context_length", 8192) * 0.75)}
AUTO_SUMMARIZE_THRESHOLD={int(GROK_MODELS_CONFIG.get(DEFAULT_GROK_MODEL, {}).get("context_length", 8192) * 0.5)}

# Performance settings for Grok
MAX_CONCURRENT_DOCS=2 # Example, adjust based on rate limits and system capacity
BATCH_SIZE=5 # Example
RATE_LIMIT_PER_MINUTE={int(RATE_LIMITS["requests_per_minute"])}
"""
