"""UI-agnostic interface definitions."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    runtime_checkable,
)

from pydantic import BaseModel, Field

from llmling_agent.models import TokenUsage  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

    from llmling_agent.commands.completion import CompletionProvider


class MessageMetadata(BaseModel):
    """Metadata for chat messages."""

    timestamp: datetime | None = Field(default=None)
    model: str | None = Field(default=None)
    token_usage: TokenUsage | None = Field(default=None)
    cost: float | None = Field(default=None)
    tool: str | None = Field(default=None)

    model_config = {"frozen": True}


class ChatMessage(BaseModel):
    """Common message format."""

    content: str
    role: Literal["user", "assistant", "system"]
    metadata: MessageMetadata | None = Field(default=None)

    model_config = {"frozen": True}


@runtime_checkable
class CoreUI(Protocol):
    """Core UI functionality required for basic chat."""

    async def send_message(
        self,
        message: ChatMessage,
        *,
        stream: bool = False,
    ) -> AsyncIterator[ChatMessage] | ChatMessage:
        """Send message to UI."""
        ...

    async def update_status(self, message: str) -> None:
        """Update UI status."""
        ...

    async def show_error(self, message: str) -> None:
        """Show error message."""
        ...


@runtime_checkable
class CompletionUI(Protocol):
    """UI supporting command completion."""

    def set_completer(self, completer: CompletionProvider | None) -> None:
        """Set completion provider."""
        ...


@runtime_checkable
class CodeEditingUI(Protocol):
    """UI supporting code editing."""

    async def edit_code(
        self,
        initial_text: str = "",
        *,
        syntax: str = "python",
        message: str | None = None,
    ) -> str:
        """Open code editor."""
        ...


@runtime_checkable
class ToolAwareUI(Protocol):
    """UI that can display tool states."""

    async def update_tool_states(
        self,
        states: dict[str, bool],
    ) -> None:
        """Update tool enable/disable states."""
        ...

    async def update_model_info(
        self,
        model: str | None = None,
        token_count: int | None = None,
        cost: float | None = None,
    ) -> None:
        """Update model usage information."""
        ...


class UserInterface(CoreUI, CompletionUI, CodeEditingUI, ToolAwareUI, Protocol):
    """Complete UI interface supporting all features."""
