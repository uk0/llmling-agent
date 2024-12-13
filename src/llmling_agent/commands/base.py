"""Base interfaces for the command system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import shlex
from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from llmling_agent.chat_session.base import AgentChatSession


class OutputWriter(Protocol):
    """Interface for command output."""

    async def print(self, message: str) -> None:
        """Write a message to output."""
        ...


@dataclass
class CommandContext:
    """Context passed to command handlers."""

    output: OutputWriter
    session: AgentChatSession
    metadata: dict[str, Any]  # For interface-specific data


@dataclass
class ParsedCommandArgs:
    """Arguments parsed from a command string."""

    args: list[str]
    kwargs: dict[str, str]


@dataclass
class ParsedCommand:
    """Complete parsed command."""

    name: str
    args: ParsedCommandArgs


type ExecuteFunc = Callable[[CommandContext, list[str], dict[str, str]], Awaitable[None]]


class CommandError(Exception):
    """Base exception for command-related errors."""


class BaseCommand(ABC):
    """Abstract base class for commands."""

    def __init__(
        self,
        name: str,
        description: str,
        category: str = "general",
        usage: str | None = None,
        help_text: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.category = category
        self.usage = usage
        self._help_text = help_text

    def format_usage(self) -> str | None:
        """Format usage string."""
        if not self.usage:
            return None
        return f"Usage: /{self.name} {self.usage}"

    @property
    def help_text(self) -> str:
        """Get help text, falling back to description if not set."""
        return self._help_text or self.description

    @abstractmethod
    async def execute(
        self,
        ctx: CommandContext,
        args: list[str],
        kwargs: dict[str, str],
    ) -> None:
        """Execute the command with parsed arguments."""
        ...


class Command(BaseCommand):
    """Concrete command that can be created directly."""

    def __init__(
        self,
        name: str,
        description: str,
        execute_func: ExecuteFunc,
        category: str = "general",
        usage: str | None = None,
        help_text: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            category=category,
            usage=usage,
            help_text=help_text,
        )
        self._execute_func = execute_func

    async def execute(
        self,
        ctx: CommandContext,
        args: list[str],
        kwargs: dict[str, str],
    ) -> None:
        """Execute the command using provided function."""
        await self._execute_func(ctx, args, kwargs)


def parse_command(cmd_str: str) -> ParsedCommand:
    """Parse command string into name and arguments.

    Args:
        cmd_str: Command string without leading slash

    Returns:
        Parsed command with name and arguments

    Raises:
        CommandError: If command syntax is invalid
    """
    try:
        parts = shlex.split(cmd_str)
    except ValueError as e:
        msg = f"Invalid command syntax: {e}"
        raise CommandError(msg) from e

    if not parts:
        msg = "Empty command"
        raise CommandError(msg)

    name = parts[0]
    args = []
    kwargs = {}

    i = 1
    while i < len(parts):
        part = parts[i]
        if part.startswith("--"):
            if i + 1 < len(parts):
                kwargs[part[2:]] = parts[i + 1]
                i += 2
            else:
                msg = f"Missing value for argument: {part}"
                raise CommandError(msg)
        else:
            args.append(part)
            i += 1

    return ParsedCommand(
        name=name,
        args=ParsedCommandArgs(args=args, kwargs=kwargs),
    )
