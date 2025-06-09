"""
Grok-3-Mini Specific Configuration and Setup

This module provides optimized configuration for using xAI's Grok-3-Mini model
as the primary LLM provider for the Legal AI System.

Reference: https://console.x.ai/team/b10c9f9e-81c9-4524-bc1c-32f6a6651671/models?modelName=grok-3-mini
"""

from typing import Dict, Any
from .settings import LegalAISettings
from ..core.llm_providers import LLMConfig, LLMProvider

# Grok model configurations
GROK_MODELS_CONFIG = {
    "grok-3-mini": {
        "model_name": "grok-3-mini",
        "context_length": 8192,
        "max_tokens": 4096,
        "temperature": 0.7,
        "use_case": "Fast, efficient legal analysis",
        "reasoning": False
    },
    "grok-3-reasoning": {
        "model_name": "grok-3-reasoning",
        "context_length": 8192,
        "max_tokens": 4096,
        "temperature": 0.7,
        "use_case": "Complex legal reasoning and analysis",
        "reasoning": True
    },
    "grok-2-1212": {
        "model_name": "grok-2-1212",
        "context_length": 8192,
        "max_tokens": 4096,
        "temperature": 0.7,
        "use_case": "Balanced performance and reasoning",
        "reasoning": False
    }
}

# Default configuration (can be switched at runtime)
DEFAULT_GROK_MODEL = "grok-3-mini"

# Optimized parameters for legal text (applies to all models)
LEGAL_ANALYSIS_PARAMS = {
    "legal_temperature": 0.3,  # Lower temperature for legal analysis
    "citation_temperature": 0.1,  # Very low for citation formatting
    "summary_temperature": 0.5,  # Moderate for summaries
    "reasoning_temperature": 0.2,  # Low for step-by-step reasoning
}

# Rate limiting (adjust based on your xAI plan)
RATE_LIMITS = {
    "requests_per_minute": 60,
    "tokens_per_minute": 100000,
    "retry_attempts": 3,
    "retry_delay": 1.0,
    "timeout": 60,
}

def create_grok_config(
    model_name: str = None, 
    api_key: str = None, 
    base_url: str = "https://api.x.ai/v1"
) -> LLMConfig:
    """
    Create optimized LLMConfig for any Grok model
    
    Args:
        model_name: Grok model name (grok-3-mini, grok-3-reasoning, grok-2-1212)
        api_key: Your xAI API key
        base_url: xAI API endpoint (default: https://api.x.ai/v1)
    
    Returns:
        LLMConfig optimized for the specified Grok model
    """
    if not model_name:
        model_name = DEFAULT_GROK_MODEL
    
    if model_name not in GROK_MODELS_CONFIG:
        raise ValueError(f"Unknown Grok model: {model_name}. Available: {list(GROK_MODELS_CONFIG.keys())}")
    
    model_config = GROK_MODELS_CONFIG[model_name]
    
    return LLMConfig(
        provider=LLMProvider.XAI,
        model=model_config["model_name"],
        api_key=api_key,
        base_url=base_url,
        temperature=model_config["temperature"],
        max_tokens=model_config["max_tokens"],
        timeout=RATE_LIMITS["timeout"],
        retry_attempts=RATE_LIMITS["retry_attempts"],
        retry_delay=RATE_LIMITS["retry_delay"]
    )

# Backward compatibility
def create_grok_3_mini_config(api_key: str, base_url: str = "https://api.x.ai/v1") -> LLMConfig:
    """Backward compatibility function for Grok-3-Mini"""
    return create_grok_config("grok-3-mini", api_key, base_url)

def get_grok_3_mini_settings() -> Dict[str, Any]:
    """
    Get environment settings optimized for Grok-3-Mini as primary provider
    
    Returns:
        Dictionary of settings for LegalAISettings
    """
    return {
        # Primary LLM Provider
        "llm_provider": "xai",
        "llm_model": "grok-3-mini",
        "llm_temperature": GROK_3_MINI_CONFIG["temperature"],
        "llm_max_tokens": GROK_3_MINI_CONFIG["max_tokens"],
        
        # xAI specific settings
        "xai_model": "grok-3-mini",
        "xai_base_url": "https://api.x.ai/v1",
        
        # Fallback to Ollama for privacy-sensitive operations
        "fallback_provider": "ollama",
        "fallback_model": "llama3.2",
        
        # Optimized context management for Grok-3-mini
        "max_context_tokens": 6000,  # Leave buffer for response
        "auto_summarize_threshold": 4000,
        
        # Performance optimizations
        "max_concurrent_documents": 2,  # Conservative for rate limits
        "batch_size": 5,
    }

def create_grok_3_mini_legal_settings(api_key: str) -> LegalAISettings:
    """
    Create complete LegalAISettings optimized for Grok-3-Mini
    
    Args:
        api_key: Your xAI API key
    
    Returns:
        LegalAISettings instance configured for Grok-3-Mini
    """
    settings_dict = get_grok_3_mini_settings()
    settings_dict["xai_api_key"] = api_key
    
    return LegalAISettings(**settings_dict)

# Model switching functionality
def switch_grok_model(current_config: LLMConfig, new_model: str) -> LLMConfig:
    """
    Switch to a different Grok model while preserving other settings
    
    Args:
        current_config: Current LLMConfig
        new_model: New Grok model name
    
    Returns:
        New LLMConfig with updated model
    """
    return create_grok_config(
        model_name=new_model,
        api_key=current_config.api_key,
        base_url=current_config.base_url
    )

def get_available_grok_models() -> Dict[str, Dict[str, Any]]:
    """Get list of available Grok models with their capabilities"""
    return GROK_MODELS_CONFIG.copy()

def is_reasoning_model(model_name: str) -> bool:
    """Check if a model supports reasoning capabilities"""
    if model_name in GROK_MODELS_CONFIG:
        return GROK_MODELS_CONFIG[model_name]["reasoning"]
    return False

def get_optimal_model_for_task(task_type: str) -> str:
    """
    Get the optimal Grok model for a specific task type
    
    Args:
        task_type: Type of task (reasoning, analysis, citation, summary)
    
    Returns:
        Recommended model name
    """
    task_recommendations = {
        "reasoning": "grok-3-reasoning",  # For complex legal reasoning
        "constitutional_analysis": "grok-3-reasoning",  # Deep constitutional review
        "case_analysis": "grok-3-reasoning",  # Complex case interpretation
        "violation_detection": "grok-3-reasoning",  # Multi-step violation analysis
        "citation": "grok-3-mini",  # Fast citation formatting
        "summary": "grok-3-mini",  # Quick document summaries
        "extraction": "grok-3-mini",  # Data extraction tasks
        "classification": "grok-3-mini",  # Document classification
        "general": "grok-3-mini"  # Default for general tasks
    }
    
    return task_recommendations.get(task_type, DEFAULT_GROK_MODEL)

# Grok prompt templates optimized for different models and legal analysis
GROK_PROMPTS = {
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
    """
}

def get_optimized_prompt(prompt_type: str, model_name: str = None, **kwargs) -> str:
    """
    Get optimized prompt for specific task and model
    
    Args:
        prompt_type: Type of prompt (legal_analysis, violation_detection, etc.)
        model_name: Grok model name (auto-detects reasoning capability)
        **kwargs: Variables to substitute in the prompt
    
    Returns:
        Formatted prompt string optimized for the model
    """
    # Determine if we should use reasoning variant
    use_reasoning = False
    if model_name and is_reasoning_model(model_name):
        use_reasoning = True
    
    # Try reasoning version first if available and appropriate
    if use_reasoning:
        reasoning_prompt_type = f"{prompt_type}_reasoning"
        if reasoning_prompt_type in GROK_PROMPTS:
            return GROK_PROMPTS[reasoning_prompt_type].format(**kwargs)
    
    # Fall back to standard prompt
    if prompt_type not in GROK_PROMPTS:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Available: {list(GROK_PROMPTS.keys())}")
    
    return GROK_PROMPTS[prompt_type].format(**kwargs)

def get_available_prompt_types() -> List[str]:
    """Get list of available prompt types"""
    # Remove _reasoning suffixes for main list
    base_types = set()
    for prompt_type in GROK_PROMPTS.keys():
        if prompt_type.endswith('_reasoning'):
            base_types.add(prompt_type[:-10])  # Remove '_reasoning'
        else:
            base_types.add(prompt_type)
    return sorted(list(base_types))

# Usage example for setting up Grok-3-Mini configuration
def setup_grok_3_mini_system(api_key: str) -> Dict[str, Any]:
    """
    Complete setup function for Grok-3-Mini based Legal AI System
    
    Args:
        api_key: Your xAI API key
    
    Returns:
        Dictionary with configuration and setup information
    """
    # Create optimized configuration
    llm_config = create_grok_3_mini_config(api_key)
    settings = create_grok_3_mini_legal_settings(api_key)
    
    return {
        "llm_config": llm_config,
        "settings": settings,
        "prompts": GROK_3_MINI_PROMPTS,
        "optimization_notes": [
            "Grok-3-Mini optimized for legal text analysis",
            "Context window: 8192 tokens",
            "Recommended for complex legal reasoning",
            "Lower temperature settings for legal precision",
            "Ollama fallback for sensitive documents"
        ]
    }

# Environment variables template for Grok-3-Mini setup
GROK_3_MINI_ENV_TEMPLATE = """
# Grok-3-Mini Optimized Configuration
LLM_PROVIDER=xai
LLM_MODEL=grok-3-mini
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096

# xAI Settings
XAI_API_KEY=your_xai_api_key_here
XAI_BASE_URL=https://api.x.ai/v1
XAI_MODEL=grok-3-mini

# Fallback to Ollama for privacy
FALLBACK_PROVIDER=ollama
FALLBACK_MODEL=llama3.2

# Optimized context management
MAX_CONTEXT_TOKENS=6000
AUTO_SUMMARIZE_THRESHOLD=4000

# Performance settings for Grok-3-Mini
MAX_CONCURRENT_DOCS=2
BATCH_SIZE=5
RATE_LIMIT_PER_MINUTE=60
"""