"""UI-agnostic interface definitions."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Protocol,
    runtime_checkable,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from slashed import CompletionProvider

    from llmling_agent.models.messages import ChatMessage


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

    async def update_status(self, message: str):
        """Update UI status."""
        ...

    async def show_error(self, message: str):
        """Show error message."""
        ...


@runtime_checkable
class CompletionUI(Protocol):
    """UI supporting command completion."""

    def set_completer(self, completer: CompletionProvider | None):
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
    ):
        """Update tool enable/disable states."""
        ...

    async def update_model_info(
        self,
        model: str | None = None,
        token_count: int | None = None,
        cost: float | None = None,
    ):
        """Update model usage information."""
        ...


class UserInterface(CoreUI, CompletionUI, CodeEditingUI, ToolAwareUI, Protocol):
    """Complete UI interface supporting all features."""
