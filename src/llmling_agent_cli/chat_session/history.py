from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.history import History


if TYPE_CHECKING:
    from collections.abc import Iterable

    from llmling_agent_cli.chat_session.base import AgentPoolView


class SessionHistory(History):
    """Simple history implementation using session storage."""

    def __init__(self, session: AgentPoolView):
        super().__init__()
        self.session = session

    def load_history_strings(self) -> Iterable[str]:
        """Load history strings (newest first)."""
        return self.session.get_commands()

    def store_string(self, string: str):
        """Store new command."""
        self.session.add_command(string)
