"""Token cost calculation utilities."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TypedDict

import httpx
from moka_py import Moka


class ModelCosts(TypedDict):
    """Cost information for a model."""

    input_cost_per_token: float
    output_cost_per_token: float


# Cache cost data for 1 hour, with 5 minute time-to-idle
_cost_cache: Moka[str, ModelCosts] = Moka(
    capacity=1000,
    ttl=86400,  # 24 hours
    tti=3600,  # 1 hour of inactivity
)

LITELLM_PRICES_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"


async def _fetch_costs() -> dict[str, ModelCosts]:
    """Fetch cost data from LiteLLM's GitHub."""
    async with httpx.AsyncClient() as client:
        response = await client.get(LITELLM_PRICES_URL)
        response.raise_for_status()
        data = response.json()

    # Extract just the cost information we need
    return {
        model: {
            "input_cost_per_token": info["input_cost_per_token"],
            "output_cost_per_token": info["output_cost_per_token"],
        }
        for model, info in data.items()
        if isinstance(info, dict)  # Skip sample_spec
        and "input_cost_per_token" in info
        and "output_cost_per_token" in info
    }


async def get_model_costs_async(model: str) -> ModelCosts | None:
    """Get cost information for a model (async version).

    Args:
        model: Model identifier (e.g. 'gpt-4', 'claude-2')

    Returns:
        Cost information if available, None otherwise
    """
    # Check cache first
    if costs := _cost_cache.get(model):
        return costs

    # Not in cache, fetch fresh data
    try:
        all_costs = await _fetch_costs()

        # Update cache with all costs
        for model_name, cost_info in all_costs.items():
            _cost_cache.set(model_name, cost_info)

        return all_costs.get(model)
    except Exception:  # noqa: BLE001
        return None


def get_model_costs(model: str) -> ModelCosts | None:
    """Get cost information for a model (sync version).

    This will use an existing event loop if available, otherwise creates one.

    Args:
        model: Model identifier (e.g. 'gpt-4', 'claude-2')

    Returns:
        Cost information if available, None otherwise
    """
    # Check cache first
    if costs := _cost_cache.get(model):
        return costs

    # Try to use existing event loop
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(get_model_costs_async(model))
    except RuntimeError:
        # No event loop in thread, create one
        with contextlib.suppress(Exception):
            return asyncio.run(get_model_costs_async(model))
    return None


def calculate_prompt_cost(prompt: str, model: str) -> float:
    """Calculate cost for prompt tokens."""
    if costs := get_model_costs(model):
        # Simple approximation: 4 chars = 1 token
        token_count = len(prompt) // 4
        return token_count * costs["input_cost_per_token"]
    return 0.0


def calculate_completion_cost(completion: str, model: str) -> float:
    """Calculate cost for completion tokens."""
    if costs := get_model_costs(model):
        # Simple approximation: 4 chars = 1 token
        token_count = len(completion) // 4
        return token_count * costs["output_cost_per_token"]
    return 0.0


async def calculate_prompt_cost_async(prompt: str, model: str) -> float:
    """Calculate cost for prompt tokens (async version)."""
    if costs := await get_model_costs_async(model):
        token_count = len(prompt) // 4
        return token_count * costs["input_cost_per_token"]
    return 0.0


async def calculate_completion_cost_async(completion: str, model: str) -> float:
    """Calculate cost for completion tokens (async version)."""
    if costs := await get_model_costs_async(model):
        token_count = len(completion) // 4
        return token_count * costs["output_cost_per_token"]
    return 0.0
