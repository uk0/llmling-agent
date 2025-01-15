"""Common types for web interface."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from llmling_agent.common_types import MessageRole  # noqa: TC001


class GradioChatMessage(TypedDict):
    """Single chat message format for Gradio chatbot."""

    content: str
    role: MessageRole
    name: NotRequired[str]
    avatar: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]


type ChatHistory = list[GradioChatMessage]
