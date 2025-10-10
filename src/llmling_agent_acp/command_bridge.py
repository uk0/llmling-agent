"""Command bridge for converting slashed commands to ACP format."""

from __future__ import annotations

import inspect
import re
from typing import TYPE_CHECKING, Any

from acp.schema import AvailableCommand, AvailableCommandInput, CommandInputHint
from llmling_agent.log import get_logger
from llmling_agent_acp.converters import to_session_updates
from llmling_agent_acp.mcp_commands import MCPPromptCommand


if TYPE_CHECKING:
    from slashed import CommandStore


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from slashed import BaseCommand, CommandContext, CommandStore

    from acp.schema import SessionNotification
    from llmling_agent.agent.context import AgentContext
    from llmling_agent_acp.session import ACPSession

logger = get_logger(__name__)
SLASH_PATTERN = re.compile(r"^/(\w+)(?:\s+(.*))?$")


class ACPOutputWriter:
    """OutputWriter that converts command output to ACP session updates."""

    def __init__(self, session_id: str) -> None:
        """Initialize with session ID for notifications."""
        self.session_id = session_id
        self.output_buffer: list[str] = []

    async def print(self, message: str = "", **kwargs: Any) -> None:
        """Capture print output."""
        self.output_buffer.append(message)

    async def write(self, text: str) -> None:
        """Capture write output."""
        self.output_buffer.append(text)

    def get_session_updates(self) -> list[SessionNotification]:
        """Convert captured output to session updates."""
        if not self.output_buffer:
            return []

        # Combine all output
        combined_output = "\n".join(self.output_buffer)
        return to_session_updates(combined_output, self.session_id)

    def clear(self) -> None:
        """Clear output buffer."""
        self.output_buffer.clear()


class ACPCommandBridge:
    """Converts slashed commands to ACP AvailableCommand format."""

    def __init__(self, command_store: CommandStore) -> None:
        """Initialize with existing command store.

        Args:
            command_store: The slashed CommandStore containing available commands
        """
        self.command_store = command_store
        self._update_callbacks: list[Callable[[], None]] = []
        self._mcp_prompt_commands: dict[str, MCPPromptCommand] = {}

    def to_available_commands(self, context: AgentContext[Any]) -> list[AvailableCommand]:
        """Convert slashed commands to ACP format.

        Args:
            context: Optional agent context to filter commands

        Returns:
            List of ACP AvailableCommand objects
        """
        commands = [  # Add regular slashed commands
            _convert_command(cmd)
            for cmd in self.command_store.list_commands()
            if _convert_command(cmd) is not None
        ]

        commands.extend([  # Add MCP prompt commands
            mcp_cmd.to_available_command()
            for mcp_cmd in self._mcp_prompt_commands.values()
        ])

        return commands

    async def execute_slash_command(
        self,
        command_text: str,
        session: ACPSession,
    ) -> AsyncIterator[SessionNotification]:
        """Execute slash command and stream results as ACP notifications.

        Args:
            command_text: Full command text (including slash)
            session: ACP session context

        Yields:
            SessionNotification objects with command output
        """
        if match := SLASH_PATTERN.match(command_text.strip()):
            command_name = match.group(1)
            args = match.group(2) or ""
            parsed = command_name, args.strip()
        else:
            parsed = None
        if not parsed:
            logger.warning("Invalid slash command: %s", command_text)
            return

        command_name, args = parsed
        output_writer = ACPOutputWriter(session.session_id)

        try:
            # Check if it's an MCP prompt command first
            if command_name in self._mcp_prompt_commands:
                mcp_cmd = self._mcp_prompt_commands[command_name]
                async for update in mcp_cmd.execute(args, session):
                    yield update
                return

            # Create command context from session
            cmd_context = self._create_command_context(session, output_writer)
            command_str = f"{command_name} {args}".strip()
            await self.command_store.execute_command(command_str, cmd_context)

            # Stream output as session updates
            for update in output_writer.get_session_updates():
                yield update

        except Exception as e:
            logger.exception("Command execution failed")
            for update in to_session_updates(f"Command error: {e}", session.session_id):
                yield update

    def _create_command_context(
        self,
        session: ACPSession,
        output_writer: ACPOutputWriter,
    ) -> CommandContext:
        """Create command context from ACP session.

        Args:
            session: ACP session
            output_writer: Output writer for command results

        Returns:
            CommandContext for slashed command execution
        """
        return self.command_store.create_context(
            data=session.agent.context,
            output_writer=output_writer,
        )

    def register_update_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for command updates.

        Args:
            callback: Function to call when commands are updated
        """
        self._update_callbacks.append(callback)

    def add_mcp_prompt_commands(self, mcp_prompts: list[Any]) -> None:
        """Add MCP prompts as slash commands.

        Args:
            mcp_prompts: List of MCP prompt objects from MCP servers
        """
        from mcp.types import Prompt as MCPPrompt

        # Clear existing MCP commands
        self._mcp_prompt_commands.clear()

        # Add new MCP prompt commands
        for prompt in mcp_prompts:
            if isinstance(prompt, MCPPrompt):
                cmd = MCPPromptCommand(prompt)
                self._mcp_prompt_commands[prompt.name] = cmd

        # Notify about command updates
        self._notify_command_update()

    def _notify_command_update(self) -> None:
        """Notify all registered callbacks about command updates."""
        for callback in self._update_callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Command update callback failed")


def is_slash_command(text: str) -> bool:
    """Check if text starts with a slash command.

    Args:
        text: Text to check

    Returns:
        True if text is a slash command
    """
    return bool(SLASH_PATTERN.match(text.strip()))


def _create_input_spec(command: BaseCommand) -> AvailableCommandInput | None:
    """Create input specification for command parameters.

    Args:
        command: Slashed command

    Returns:
        Input specification or None if no parameters
    """
    # For now, create a simple text input hint
    # This could be enhanced to parse actual parameter signatures
    try:
        sig = inspect.signature(command.execute)
        params = [n for n, _ in sig.parameters.items() if n not in {"self", "ctx"}]
        if params:
            hint = f"Parameters: {', '.join(params)}"
            return AvailableCommandInput(root=CommandInputHint(hint=hint))
    except Exception:  # noqa: BLE001
        pass

    return None


def _convert_command(command: BaseCommand) -> AvailableCommand:
    """Convert a single slashed command to ACP format.

    Args:
        command: Slashed command to convert

    Returns:
        ACP AvailableCommand or None if conversion fails
    """
    description = command.description
    spec = _create_input_spec(command)
    return AvailableCommand(name=command.name, description=description, input=spec)
