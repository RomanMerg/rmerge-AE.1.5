"""Track and estimate API costs using OpenRouter pricing."""

# Approximate pricing per 1M tokens (as of 2025, update from OpenRouter API later)
# These are placeholder values — Session 4 will fetch real-time pricing
MODEL_PRICING = {
    "openai/gpt-5-nano": {"input": 0.10, "output": 0.40},
    "openai/gpt-5-mini": {"input": 0.40, "output": 1.60},
    "openai/gpt-5": {"input": 2.00, "output": 8.00},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD for a single API call."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0

    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def format_cost(cost_usd: float) -> str:
    """Format cost for display."""
    if cost_usd < 0.001:
        return f"${cost_usd:.6f}"
    return f"${cost_usd:.4f}"


class SessionCostTracker:
    """Track cumulative costs for a session."""

    def __init__(self):
        self.total_cost = 0.0
        self.calls = []

    def add(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        self.total_cost += cost
        self.calls.append({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
        })
        return cost

    def summary(self) -> str:
        return f"Session cost: {format_cost(self.total_cost)} ({len(self.calls)} API calls)"
