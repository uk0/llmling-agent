"""Utilities for working with pydantic-ai types and objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict


if TYPE_CHECKING:
    from pydantic_ai.result import Cost


class TokenUsage(TypedDict):
    """Token usage statistics from model responses."""

    total: int
    """Total tokens used"""
    prompt: int
    """Tokens used in the prompt"""
    completion: int
    """Tokens used in the completion"""


def extract_token_usage(cost: Cost) -> TokenUsage | None:
    """Extract token usage statistics from a cost object.

    Args:
        cost: Cost object from model response

    Returns:
        Token usage statistics if available, None otherwise
    """
    if (
        cost
        and cost.total_tokens is not None
        and cost.request_tokens is not None
        and cost.response_tokens is not None
    ):
        return TokenUsage(
            total=cost.total_tokens,
            prompt=cost.request_tokens,
            completion=cost.response_tokens,
        )
    return None
