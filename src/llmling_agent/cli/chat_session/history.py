"""Interactive chat session implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.history import History


if TYPE_CHECKING:
    from collections.abc import Iterable

    from llmling_agent.chat_session.base import AgentChatSession


class SessionHistory(History):
    """History implementation backed by AgentChatSession."""

    def __init__(self, session: AgentChatSession) -> None:
        super().__init__()
        self.session = session

    def load_history_strings(self) -> Iterable[str]:
        """Load history strings from session."""
        return self.session.get_commands()

    def store_string(self, string: str) -> None:
        """Store string in session history."""
        self.session.add_command(string)
