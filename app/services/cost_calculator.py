"""Cost calculator for LLM API usage."""

from app.config import settings
from app.models import CostBreakdown


# Pricing per 1M tokens (as of early 2024 - update as needed)
# Format: {provider: {model: {"input": price, "output": price}}}
MODEL_PRICING = {
    "groq": {
        # Groq has generous free tier, minimal cost after
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
        "llama3-8b-8192": {"input": 0.05, "output": 0.08},
        "llama3-70b-8192": {"input": 0.59, "output": 0.79},
        "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
        "gemma-7b-it": {"input": 0.07, "output": 0.07},
    },
    "google": {
        # Gemini pricing
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-pro": {"input": 0.50, "output": 1.50},
    },
    "openai": {
        # OpenAI pricing
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    },
    "anthropic": {
        # Anthropic pricing
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    },
}


def get_model_pricing(provider: str, model: str) -> dict:
    """
    Get pricing for a model.
    
    Args:
        provider: Provider name (groq, google, openai, anthropic)
        model: Model name
        
    Returns:
        Dict with 'input' and 'output' prices per 1M tokens
    """
    provider_pricing = MODEL_PRICING.get(provider, {})
    
    # Try exact match
    if model in provider_pricing:
        return provider_pricing[model]
    
    # Try partial match (model names can vary)
    for model_name, pricing in provider_pricing.items():
        if model_name in model or model in model_name:
            return pricing
    
    # Default fallback pricing
    return {"input": 1.0, "output": 2.0}


def calculate_cost(
    tier1_tokens: int,
    tier2_tokens: int,
    tier1_provider: str = None,
    tier1_model: str = None,
    tier2_provider: str = None,
    tier2_model: str = None,
) -> CostBreakdown:
    """
    Calculate cost breakdown for a query.
    
    Args:
        tier1_tokens: Total tokens used by Tier-1 model
        tier2_tokens: Total tokens used by Tier-2 model
        tier1_provider: Tier-1 provider (default from settings)
        tier1_model: Tier-1 model name (default from settings)
        tier2_provider: Tier-2 provider (default from settings)
        tier2_model: Tier-2 model name (default from settings)
        
    Returns:
        CostBreakdown with detailed cost information
    """
    # Use settings if not provided
    tier1_provider = tier1_provider or settings.tier1_provider
    tier1_model = tier1_model or settings.tier1_model
    tier2_provider = tier2_provider or settings.tier2_provider
    tier2_model = tier2_model or settings.tier2_model
    
    # Get pricing
    tier1_pricing = get_model_pricing(tier1_provider, tier1_model)
    tier2_pricing = get_model_pricing(tier2_provider, tier2_model)
    
    # Assume 50/50 split between input and output tokens (rough estimate)
    # In reality, you'd track input vs output separately
    tier1_input = tier1_tokens * 0.6  # Classification prompts are mostly input
    tier1_output = tier1_tokens * 0.4
    
    tier2_input = tier2_tokens * 0.4  # Synthesis has more output
    tier2_output = tier2_tokens * 0.6
    
    # Calculate costs (prices are per 1M tokens)
    tier1_cost = (
        (tier1_input / 1_000_000) * tier1_pricing["input"] +
        (tier1_output / 1_000_000) * tier1_pricing["output"]
    )
    
    tier2_cost = (
        (tier2_input / 1_000_000) * tier2_pricing["input"] +
        (tier2_output / 1_000_000) * tier2_pricing["output"]
    )
    
    total_cost = tier1_cost + tier2_cost
    
    # Calculate what it would cost if we used Tier-2 for everything
    total_tokens = tier1_tokens + tier2_tokens
    tier2_only_cost = (
        (total_tokens * 0.5 / 1_000_000) * tier2_pricing["input"] +
        (total_tokens * 0.5 / 1_000_000) * tier2_pricing["output"]
    )
    
    savings = tier2_only_cost - total_cost
    savings_percent = (savings / tier2_only_cost * 100) if tier2_only_cost > 0 else 0
    
    return CostBreakdown(
        tier1_tokens=tier1_tokens,
        tier2_tokens=tier2_tokens,
        tier1_cost_usd=round(tier1_cost, 6),
        tier2_cost_usd=round(tier2_cost, 6),
        total_cost_usd=round(total_cost, 6),
        tier1_model=f"{tier1_provider}/{tier1_model}",
        tier2_model=f"{tier2_provider}/{tier2_model}",
        savings_vs_tier2_only=round(savings_percent, 1),
    )


def format_cost(cost_usd: float) -> str:
    """Format cost for display."""
    if cost_usd < 0.0001:
        return "< $0.0001"
    elif cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    elif cost_usd < 1:
        return f"${cost_usd:.3f}"
    else:
        return f"${cost_usd:.2f}"
