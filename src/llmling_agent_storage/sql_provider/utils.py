"""Utilities for database storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from llmling_agent.models.messages import ChatMessage


if TYPE_CHECKING:
    from llmling_agent_storage.sql_provider.models import Message


def db_message_to_pydantic_ai_message(msg: Message) -> ModelMessage:
    """Convert a database message to a pydantic-ai message."""
    match msg.role:
        case "user":
            return ModelRequest(parts=[UserPromptPart(content=msg.content)])
        case "assistant":
            return ModelResponse(parts=[TextPart(content=msg.content)])
        case "system":
            return ModelRequest(parts=[SystemPromptPart(content=msg.content)])
    error_msg = f"Unknown message role: {msg.role}"
    raise ValueError(error_msg)


def db_message_to_chat_message(msg: Message) -> ChatMessage[str]:
    """Convert database message to ChatMessage format."""
    from llmling_agent.models.messages import TokenCost

    # Create cost info if we have token usage
    cost_info = None
    if msg.total_tokens is not None:
        cost_info = TokenCost(
            token_usage={
                "total": msg.total_tokens,
                "prompt": msg.prompt_tokens or 0,
                "completion": msg.completion_tokens or 0,
            },
            total_cost=msg.cost or 0.0,
        )

    return ChatMessage[str](
        content=msg.content,
        role=msg.role,  # type: ignore
        name=msg.name,
        model=msg.model,
        cost_info=cost_info,
        response_time=msg.response_time,
        forwarded_from=msg.forwarded_from or [],
        timestamp=msg.timestamp,
    )


def parse_model_info(model: str | None) -> tuple[str | None, str | None]:
    """Parse model string into provider and name.

    Args:
        model: Full model string (e.g., "openai:gpt-4", "anthropic/claude-2")

    Returns:
        Tuple of (provider, name)
    """
    if not model:
        return None, None

    # Try splitting by ':' or '/'
    parts = model.split(":") if ":" in model else model.split("/")

    if len(parts) == 2:  # noqa: PLR2004
        provider, name = parts
        return provider.lower(), name

    # No provider specified, try to infer
    name = parts[0]
    if name.startswith(("gpt-", "text-", "dall-e")):
        return "openai", name
    if name.startswith("claude"):
        return "anthropic", name
    if name.startswith(("llama", "mistral")):
        return "meta", name

    return None, name
