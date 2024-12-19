"""Token cost calculation utilities."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, TypedDict

import diskcache
import httpx
from platformdirs import user_data_dir

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.models.messages import TokenUsage


logger = get_logger(__name__)


class ModelCosts(TypedDict):
    """Cost information for a model."""

    input_cost_per_token: float
    output_cost_per_token: float


# Cache cost data persistently
PRICING_DIR = pathlib.Path(user_data_dir("llmling", "llmling")) / "pricing"
PRICING_DIR.mkdir(parents=True, exist_ok=True)
_cost_cache = diskcache.Cache(directory=str(PRICING_DIR))

# Cache timeout in seconds (24 hours)
_CACHE_TIMEOUT = 86400

LITELLM_PRICES_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"


def find_litellm_model_name(model: str) -> str | None:
    """Find matching model name in LiteLLM pricing data.

    Args:
        model: Input model name (e.g. "openai:gpt-4", "gpt-4")

    Returns:
        Matching LiteLLM model name if found, None otherwise
    """
    logger.debug("Looking up model costs for: %s", model)

    # Normalize case
    model = model.lower()

    # First check direct match
    if model in _cost_cache:
        logger.debug("Found direct cache match for: %s", model)
        return model

    # For provider:model format, try both variants
    if ":" in model:
        provider, model_name = model.split(":", 1)
        # Try just model name (normalized)
        model_name = model_name.lower()
        if _cost_cache.get(model_name, None) is not None:
            logger.debug("Found cache match for base name: %s", model_name)
            return model_name
        # Try provider/model format (normalized)
        provider_format = f"{provider.lower()}/{model_name}"
        if _cost_cache.get(provider_format, None) is not None:
            logger.debug("Found cache match for provider format: %s", provider_format)
            return provider_format

    logger.debug("No cache match found for: %s", model)
    return None


async def get_model_costs(model: str) -> ModelCosts | None:
    """Get cost information for a model."""
    # Find matching model name in LiteLLM format
    if litellm_name := find_litellm_model_name(model):
        return _cost_cache.get(litellm_name)

    # Not in cache, try to fetch
    try:
        logger.debug("Downloading pricing data from LiteLLM...")
        async with httpx.AsyncClient() as client:
            response = await client.get(LITELLM_PRICES_URL)
            response.raise_for_status()
            data = response.json()
        logger.debug("Successfully downloaded pricing data")

        # Extract just the cost information we need
        all_costs: dict[str, ModelCosts] = {}
        for name, info in data.items():
            if not isinstance(info, dict):  # Skip sample_spec
                continue
            if "input_cost_per_token" not in info or "output_cost_per_token" not in info:
                continue
            # Store with normalized case
            all_costs[name.lower()] = ModelCosts(
                input_cost_per_token=float(info["input_cost_per_token"]),
                output_cost_per_token=float(info["output_cost_per_token"]),
            )

        logger.debug("Extracted costs for %d models", len(all_costs))

        # Update cache with all costs
        for model_name, cost_info in all_costs.items():
            _cost_cache.set(model_name, cost_info, expire=_CACHE_TIMEOUT)
        logger.debug("Updated cache with new pricing data")

        # Return costs for requested model
        if model in all_costs:
            logger.debug("Found costs for requested model: %s", model)
            return all_costs[model]
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to get model costs: %s", e)
        return None
    else:
        logger.debug("No costs found for model: %s", model)
        return None


async def calculate_token_cost(
    model: str,
    token_usage: TokenUsage,
) -> float | None:
    """Calculate total cost for token usage."""
    costs = await get_model_costs(model)
    if costs:
        prompt_cost = token_usage["prompt"] * costs["input_cost_per_token"]
        completion_cost = token_usage["completion"] * costs["output_cost_per_token"]
        total_cost = float(prompt_cost + completion_cost)
        logger.debug(
            "Cost calculation - prompt: $%.6f, completion: $%.6f, total: $%.6f",
            prompt_cost,
            completion_cost,
            total_cost,
        )
        return total_cost
    logger.debug("No costs found for model")
    return None
