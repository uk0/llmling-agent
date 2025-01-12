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
