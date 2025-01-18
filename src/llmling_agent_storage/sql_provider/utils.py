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
