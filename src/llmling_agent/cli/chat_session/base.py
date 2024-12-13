from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from rich.console import Console

    from llmling_agent.chat_session import AgentChatSession
    from llmling_agent.cli.chat_session.config import SessionState


@dataclass
class CommandContext:
    """Context passed to command handlers."""

    console: Console
    session: AgentChatSession
    state: SessionState
    args: str


class Command(Protocol):
    """Command interface."""

    name: str
    description: str
    usage: str | None

    async def execute(self, ctx: CommandContext) -> None:
        """Execute command."""
        ...

    def format_usage(self) -> str | None:
        """Get formatted usage string."""
        ...


class BaseCommand:
    """Base command implementation."""

    def __init__(
        self,
        name: str,
        description: str,
        usage: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.usage = usage

    def format_usage(self) -> str | None:
        """Get formatted usage string."""
        if not self.usage:
            return None
        return f"Usage: /{self.name} {self.usage}"

    @abstractmethod
    async def execute(self, ctx: CommandContext) -> None:
        """Execute command."""
        ...
