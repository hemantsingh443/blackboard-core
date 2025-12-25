"""
Pricing and Cost Management for Blackboard SDK

Provides LLM cost estimation and budget enforcement.

Strategy: "Delegate, then Override"
- Uses LiteLLM as the primary pricing source (community-maintained)
- Allows user-configurable overrides for custom contracts

Usage:
    from blackboard.pricing import get_model_cost, estimate_cost

    # Get cost per 1k tokens
    input_cost, output_cost = get_model_cost("gpt-4")

    # Estimate cost for a response
    cost = estimate_cost("gpt-4", input_tokens=500, output_tokens=200)
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

logger = logging.getLogger("blackboard.pricing")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelPricing:
    """Pricing per 1,000 tokens for a model."""
    input_per_1k: float
    output_per_1k: float


# =============================================================================
# Default Pricing (fallback when LiteLLM unavailable)
# =============================================================================

# Prices as of Dec 2025 (USD per 1k tokens)
DEFAULT_PRICING: Dict[str, ModelPricing] = {
    # OpenAI – legacy models kept, plus newer ones
    "gpt-4": ModelPricing(0.005, 0.03),          # $5 / $30 per 1M (legacy)
    "gpt-4-turbo": ModelPricing(0.005, 0.03),    # gpt-4-turbo-2024-04-09 $5 / $30 per 1M
    "gpt-4o": ModelPricing(0.0025, 0.01),        # $2.50 / $10 per 1M
    "gpt-4o-mini": ModelPricing(0.00015, 0.0006),# $0.15 / $0.60 per 1M
    "gpt-3.5-turbo": ModelPricing(0.0005, 0.0015),# $0.50 / $1.50 per 1M (legacy)

    # Newer OpenAI series (o- and gpt-5- series)
    "gpt-5": ModelPricing(0.00125, 0.01),        # $1.25 / $10 per 1M
    "gpt-5-mini": ModelPricing(0.00025, 0.002),  # $0.25 / $2 per 1M
    "gpt-5-nano": ModelPricing(0.00005, 0.0004), # $0.05 / $0.40 per 1M

    "gpt-4.1": ModelPricing(0.002, 0.008),       # $2 / $8 per 1M
    "gpt-4.1-mini": ModelPricing(0.0004, 0.0016),# $0.40 / $1.60 per 1M
    "gpt-4.1-nano": ModelPricing(0.0001, 0.0004),# $0.10 / $0.40 per 1M

    "o1": ModelPricing(0.015, 0.06),             # $15 / $60 per 1M (flex tier)
    "o1-pro": ModelPricing(0.15, 0.6),           # $150 / $600 per 1M
    "o1-mini": ModelPricing(0.0011, 0.0044),     # $1.10 / $4.40 per 1M

    "o3": ModelPricing(0.002, 0.008),            # $2 / $8 per 1M
    "o3-pro": ModelPricing(0.02, 0.08),          # $20 / $80 per 1M
    "o3-mini": ModelPricing(0.0011, 0.0044),     # $1.10 / $4.40 per 1M

    "o4-mini": ModelPricing(0.0011, 0.0044),     # $1.10 / $4.40 per 1M

    # Anthropic – Claude 3.x pricing (per 1M -> per 1K)
    "claude-3-opus": ModelPricing(0.015, 0.075),     # $15 / $75 per 1M
    "claude-3-sonnet": ModelPricing(0.003, 0.015),   # $3 / $15 per 1M
    "claude-3-haiku": ModelPricing(0.00025, 0.00125),# $0.25 / $1.25 per 1M

    "claude-3-5-sonnet": ModelPricing(0.003, 0.015), # $3 / $15 per 1M
    "claude-3-5-haiku": ModelPricing(0.001, 0.005),  # $1 / $5 per 1M

    # Google Gemini – 1.x, 2.x and 3.x family
    "gemini-pro": ModelPricing(0.00125, 0.01),       # legacy Gemini 1.5 Pro $1.25 / $10 per 1M
    "gemini-1.5-pro": ModelPricing(0.00125, 0.01),   # $1.25 / $10 per 1M
    "gemini-1.5-flash": ModelPricing(0.0003, 0.0025),# $0.30 / $2.50 per 1M

    "gemini-2.0-flash": ModelPricing(0.0001, 0.0004),# $0.10 / $0.40 per 1M
    "gemini-2.0-flash-lite": ModelPricing(0.000075, 0.0003),# $0.075 / $0.30 per 1M
    "gemini-2.0-flash-exp": ModelPricing(0.0001, 0.0004),  # same as 2.0 Flash

    "gemini-2.5-pro": ModelPricing(0.00125, 0.01),   # $1.25 / $10 per 1M (<=200k tokens)
    "gemini-2.5-flash": ModelPricing(0.0003, 0.0025),# $0.30 / $2.50 per 1M
    "gemini-2.5-flash-lite": ModelPricing(0.0001, 0.0004),# $0.10 / $0.40 per 1M

    "gemini-3-pro-preview": ModelPricing(0.002, 0.012),    # $2 / $12 per 1M
    "gemini-3-flash-preview": ModelPricing(0.0005, 0.003), # $0.50 / $3 per 1M

    # Mistral
    "mistral-large": ModelPricing(0.0004, 0.002),    # $0.40 / $2 per 1M
    "mistral-small": ModelPricing(0.0002, 0.0006),   # $0.20 / $0.60 per 1M

    # Fallback
    "default": ModelPricing(0.001, 0.002),
}


# User-provided overrides (set via configure_pricing)
_custom_pricing: Dict[str, ModelPricing] = {}


# =============================================================================
# Configuration
# =============================================================================

def configure_pricing(
    custom_pricing: Optional[Dict[str, Tuple[float, float]]] = None,
    pricing_callback: Optional[Callable[[str], Optional[Tuple[float, float]]]] = None
) -> None:
    """
    Configure custom pricing overrides.
    
    Args:
        custom_pricing: Dict mapping model names to (input_per_1k, output_per_1k)
        pricing_callback: Optional callback for dynamic pricing lookup
        
    Example:
        configure_pricing({
            "my-custom-model": (0.01, 0.02),
            "azure/gpt-4": (0.025, 0.05),  # Azure contract rate
        })
    """
    global _custom_pricing, _pricing_callback
    
    if custom_pricing:
        for model, (input_cost, output_cost) in custom_pricing.items():
            _custom_pricing[model] = ModelPricing(input_cost, output_cost)
    
    if pricing_callback:
        _pricing_callback = pricing_callback


_pricing_callback: Optional[Callable[[str], Optional[Tuple[float, float]]]] = None


# =============================================================================
# Pricing Lookup
# =============================================================================

def get_model_cost(model: str) -> Tuple[float, float]:
    """
    Get the cost per 1,000 tokens for a model.
    
    Priority:
    1. User-configured custom pricing
    2. LiteLLM model_cost (if available)
    3. Default fallback pricing
    
    Args:
        model: Model identifier (e.g., "gpt-4", "claude-3-sonnet")
        
    Returns:
        Tuple of (input_per_1k, output_per_1k) in USD
    """
    # 1. Check custom overrides
    if model in _custom_pricing:
        p = _custom_pricing[model]
        return (p.input_per_1k, p.output_per_1k)
    
    # 2. Check callback
    if _pricing_callback:
        result = _pricing_callback(model)
        if result:
            return result
    
    # 3. Try LiteLLM model_cost
    try:
        from litellm import model_cost
        if model in model_cost:
            cost_info = model_cost[model]
            input_cost = cost_info.get("input_cost_per_token", 0) * 1000
            output_cost = cost_info.get("output_cost_per_token", 0) * 1000
            return (input_cost, output_cost)
    except (ImportError, Exception) as e:
        logger.debug(f"LiteLLM pricing lookup failed: {e}")
    
    # 4. Default pricing (try to match prefix - longest match wins)
    sorted_keys = sorted(
        [k for k in DEFAULT_PRICING.keys() if k != "default"],
        key=len,
        reverse=True  # Longest first to avoid "gpt-4" matching before "gpt-4o"
    )
    for prefix in sorted_keys:
        if model.startswith(prefix):
            pricing = DEFAULT_PRICING[prefix]
            return (pricing.input_per_1k, pricing.output_per_1k)
    
    # 5. Ultimate fallback
    fallback = DEFAULT_PRICING["default"]
    logger.warning(f"No pricing found for model '{model}', using default")
    return (fallback.input_per_1k, fallback.output_per_1k)


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int
) -> float:
    """
    Estimate the cost of an LLM call.
    
    Args:
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Estimated cost in USD
    """
    input_per_1k, output_per_1k = get_model_cost(model)
    
    input_cost = (input_tokens / 1000) * input_per_1k
    output_cost = (output_tokens / 1000) * output_per_1k
    
    return round(input_cost + output_cost, 10)  # Round to avoid float precision issues


# =============================================================================
# Budget Error
# =============================================================================

class BudgetExceededError(Exception):
    """
    Raised when accumulated cost exceeds the configured budget.
    
    Attributes:
        accumulated_cost: Total cost spent so far
        budget: The configured budget limit
        model: The model that would have been called
    """
    
    def __init__(
        self,
        accumulated_cost: float,
        budget: float,
        model: Optional[str] = None
    ):
        self.accumulated_cost = accumulated_cost
        self.budget = budget
        self.model = model
        
        message = (
            f"Budget exceeded: ${accumulated_cost:.4f} spent, "
            f"limit is ${budget:.4f}"
        )
        if model:
            message += f" (blocked call to {model})"
        
        super().__init__(message)
