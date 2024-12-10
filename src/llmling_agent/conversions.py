"""Conversion utilities between LLMling and pydantic-ai types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import messages


if TYPE_CHECKING:
    from llmling.prompts import PromptMessage


def llmling_to_pydantic_ai_message(msg: PromptMessage) -> messages.Message:
    """Convert LLMling message to pydantic-ai message.

    Args:
        msg: LLMling PromptMessage to convert

    Returns:
        Equivalent pydantic-ai Message

    Raises:
        ValueError: If message content type is not supported
    """
    match msg.content:
        case str():
            return messages.PromptMessage(role=msg.role, content=msg.content)
        case list():
            # Convert list of MessageContent to pydantic-ai format
            content = [
                messages.MessageContent(
                    type=item.type,
                    content=item.content,
                    alt_text=item.alt_text,
                )
                for item in msg.content
            ]
            return messages.PromptMessage(role=msg.role, content=content)
        case _:
            msg = f"Invalid message content type: {type(msg.content)}"
            raise ValueError(msg)


def convert_message_history(
    messages: list[PromptMessage] | None,
) -> list[messages.Message] | None:
    """Convert a list of LLMling messages to pydantic-ai messages.

    Args:
        messages: List of LLMling messages to convert

    Returns:
        List of equivalent pydantic-ai messages or None if input is None
    """
    if messages is None:
        return None
    return [llmling_to_pydantic_ai_message(msg) for msg in messages]
