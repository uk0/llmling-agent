"""Storage utiliy functions."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling_agent.pydantic_ai_utils import TokenUsage
    from llmling_agent.storage.models import Message


def aggregate_token_usage(messages: Sequence[Message]) -> TokenUsage:
    """Sum up tokens from a sequence of storage Message objects."""
    total = sum(msg.total_tokens or 0 for msg in messages)
    prompt = sum(msg.prompt_tokens or 0 for msg in messages)
    completion = sum(msg.completion_tokens or 0 for msg in messages)

    return {"total": total, "prompt": prompt, "completion": completion}
