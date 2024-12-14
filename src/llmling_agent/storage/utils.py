"""Storage utiliy functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.pydantic_ai_utils import TokenUsage


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling_agent.storage.models import Message


def aggregate_token_usage(messages: Sequence[Message]) -> TokenUsage:
    """Sum up tokens from a sequence of storage Message objects."""
    empty = TokenUsage(total=0, prompt=0, completion=0)
    total = sum((msg.token_usage or empty).get("total", 0) for msg in messages)
    comp = sum((msg.token_usage or empty).get("completion", 0) for msg in messages)
    prompt = sum((msg.token_usage or empty).get("prompt", 0) for msg in messages)
    return {"total": total, "completion": comp, "prompt": prompt}
