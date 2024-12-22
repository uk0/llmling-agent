"""Test implementation of UI protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from llmling_agent.models import ChatMessage
from llmling_agent.ui.interfaces import UserInterface


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from slashed import CompletionProvider


@dataclass
class UIInteraction:
    """Record of a UI interaction."""

    type: str
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class DummyUI(UserInterface):
    """Test implementation recording all UI interactions."""

    def __init__(
        self,
        *,
        message_responses: dict[str, str] | None = None,
        code_responses: dict[str, str] | None = None,
        raise_errors: bool = False,
    ):
        """Initialize test UI."""
        self.interactions: list[UIInteraction] = []
        self.message_responses = message_responses or {}
        self.code_responses = code_responses or {}
        self.raise_errors = raise_errors
        self.completer: CompletionProvider | None = None

    def _record(self, type_: str, **data: Any):
        """Record an interaction."""
        self.interactions.append(UIInteraction(type=type_, data=data))

    async def send_message(
        self,
        message: ChatMessage,
        *,
        stream: bool = False,
    ) -> AsyncIterator[ChatMessage] | ChatMessage:
        """Record message and return predefined response."""
        self._record("send_message", message=message, stream=stream)
        msg = f"Test response to: {message.content}"
        response = self.message_responses.get(message.content, msg)
        chat_message: ChatMessage[str] = ChatMessage(content=response, role="assistant")
        if stream:

            async def message_stream() -> AsyncIterator[ChatMessage]:
                for word in response.split():
                    yield ChatMessage(content=word + " ", role="assistant")
                yield chat_message

            return message_stream()
        return chat_message

    async def update_status(self, message: str):
        """Record status update."""
        self._record("update_status", message=message)

    async def show_error(self, message: str):
        """Record error and optionally raise."""
        self._record("show_error", message=message)
        if self.raise_errors:
            raise RuntimeError(message)

    def set_completer(self, completer: CompletionProvider | None):
        """Record completer setting."""
        self._record("set_completer", completer=completer)
        self.completer = completer

    async def edit_code(
        self,
        initial_text: str = "",
        *,
        syntax: str = "python",
        message: str | None = None,
    ) -> str:
        """Record code edit and return predefined response."""
        self._record(
            "edit_code",
            initial_text=initial_text,
            syntax=syntax,
            message=message,
        )
        return self.code_responses.get(initial_text, initial_text + "\n# Edited")

    async def update_tool_states(self, states: dict[str, bool]):
        """Record tool state update."""
        self._record("update_tool_states", states=states)

    async def update_model_info(
        self,
        model: str | None = None,
        token_count: int | None = None,
        cost: float | None = None,
    ):
        """Record model info update."""
        self._record("update_model_info", model=model, token_count=token_count, cost=cost)

    def get_interactions(
        self,
        type_: str | None = None,
    ) -> list[UIInteraction]:
        """Get recorded interactions, optionally filtered by type."""
        if type_ is None:
            return self.interactions
        return [i for i in self.interactions if i.type == type_]

    def clear_interactions(self):
        """Clear recorded interactions."""
        self.interactions.clear()
