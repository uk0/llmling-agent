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
    if not isinstance(msg, dict):
        msg = f"Chat message must be a dictionary, got {type(msg)}"
        raise ValueError(msg)  # noqa: TRY004

    if "content" not in msg or "role" not in msg:
        error = "Chat message must have 'content' and 'role' fields"
        raise ValueError(error)

    if msg["role"] not in ("user", "assistant", "system"):
        error = f"Invalid role: {msg['role']}"
        raise ValueError(error)

    return msg  # type: ignore
