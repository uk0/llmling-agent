"""Common types for web interface."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class ChatMessage(TypedDict):
    """Single chat message format for Gradio chatbot."""

    content: str
    role: Literal["user", "assistant", "system"]
    name: NotRequired[str]
    avatar: NotRequired[str]


type ChatHistory = list[ChatMessage]
