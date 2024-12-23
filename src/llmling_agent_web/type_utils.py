"""Common types for web interface."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


class GradioChatMessage(TypedDict):
    """Single chat message format for Gradio chatbot."""

    content: str
    role: Literal["user", "assistant", "system"]
    name: NotRequired[str]
    avatar: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]


type ChatHistory = list[GradioChatMessage]


def validate_chat_message(msg: Any) -> GradioChatMessage:
    """Validate chat message format.

    Args:
        msg: Message to validate

    Returns:
        Validated ChatMessage

    Raises:
        ValueError: If message format is invalid
    """
    match msg:
        case {"content": str(), "role": ("user" | "assistant" | "system")}:
            return msg
        case {"content": str(), "role": invalid_role}:
            error = f"Invalid role: {invalid_role}"
            raise ValueError(error)
        case {"content": _, "role": _}:
            error = "Content and role must be strings"
            raise ValueError(error)
        case dict():
            error = "Chat message must have 'content' and 'role' fields"
            raise ValueError(error)
        case _:
            error = f"Chat message must be a dictionary, got {type(msg)}"
            raise ValueError(error)
