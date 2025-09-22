"""Command bridge for converting slashed commands to ACP format."""

from __future__ import annotations

import inspect
import re
from typing import TYPE_CHECKING, Any

from acp.schema import AvailableCommand, AvailableCommandInput, AvailableCommandInput1

from llmling_agent.log import get_logger
from llmling_agent_acp.converters import to_session_updates


if TYPE_CHECKING:
    from slashed import CommandStore


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from acp.schema import SessionNotification
    from slashed import BaseCommand, CommandContext, CommandStore

    from llmling_agent.agent.context import AgentContext
    from llmling_agent_acp.session import ACPSession

logger = get_logger(__name__)


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
        self._slash_pattern = re.compile(r"^/(\w+)(?:\s+(.*))?$")
        self._update_callbacks: list[Callable[[], None]] = []

    def to_available_commands(
        self, context: AgentContext | None = None
    ) -> list[AvailableCommand]:
        """Convert slashed commands to ACP format.

        Args:
            context: Optional agent context to filter commands

        Returns:
            List of ACP AvailableCommand objects
        """
        commands = []

        for cmd_name, command in self.command_store._commands.items():
            # Skip internal or hidden commands
            if cmd_name.startswith("_") or getattr(command, "hidden", False):
                continue

            # Filter commands based on context/capabilities if needed
            if context and not self._is_command_available(command, context):
                continue

            acp_command = self._convert_command(command)
            if acp_command:
                commands.append(acp_command)

        return commands

    def _convert_command(self, command: BaseCommand) -> AvailableCommand | None:
        """Convert a single slashed command to ACP format.

        Args:
            command: Slashed command to convert

        Returns:
            ACP AvailableCommand or None if conversion fails
        """
        try:
            # Get command metadata
            name = command.name
            description = self._get_command_description(command)

            # Create input specification if command has parameters
            input_spec = self._create_input_spec(command)

            return AvailableCommand(name=name, description=description, input=input_spec)

        except Exception:
            msg = "Failed to convert command %s"
            logger.exception(msg, getattr(command, "name", "unknown"))
            return None

    def _get_command_description(self, command: BaseCommand) -> str:
        """Extract description from command.

        Args:
            command: Slashed command

        Returns:
            Command description
        """
        # Try various sources for description
        if command.description:
            return command.description

        if command.__doc__:
            # Use first line of docstring
            return command.__doc__.strip().split("\n")[0]

        # Fallback to command name
        return f"Execute {command.name} command"

    def _create_input_spec(self, command: BaseCommand) -> AvailableCommandInput | None:
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
            params = [
                name
                for name, param in sig.parameters.items()
                if name not in ("self", "ctx")
            ]

            if params:
                hint = f"Parameters: {', '.join(params)}"
                return AvailableCommandInput(root=AvailableCommandInput1(hint=hint))
        except Exception:  # noqa: BLE001
            pass

        return None

    def _is_command_available(self, command: BaseCommand, context: AgentContext) -> bool:
        """Check if command is available in the given context.

        Args:
            command: Command to check
            context: Agent context

        Returns:
            True if command should be available
        """
        # Basic filtering - can be enhanced
        category = getattr(command, "category", None)

        # Pool commands require pool context
        if category == "pool" and not context.pool:
            return False

        # Tool commands require tool capabilities
        if category == "tools":
            return getattr(context.capabilities, "can_access_tools", True)

        return True

    def is_slash_command(self, text: str) -> bool:
        """Check if text starts with a slash command.

        Args:
            text: Text to check

        Returns:
            True if text is a slash command
        """
        return bool(self._slash_pattern.match(text.strip()))

    def parse_slash_command(self, text: str) -> tuple[str, str] | None:
        """Parse slash command text.

        Args:
            text: Command text to parse

        Returns:
            Tuple of (command_name, args) or None if not a command
        """
        match = self._slash_pattern.match(text.strip())
        if match:
            command_name = match.group(1)
            args = match.group(2) or ""
            return command_name, args.strip()
        return None

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
        # Parse command
        parsed = self.parse_slash_command(command_text)
        if not parsed:
            logger.warning("Invalid slash command: %s", command_text)
            return

        command_name, args = parsed

        # Create output writer
        output_writer = ACPOutputWriter(session.session_id)

        try:
            # Create command context from session
            cmd_context = self._create_command_context(session, output_writer)

            # Execute command
            command_str = f"{command_name} {args}".strip()
            await self.command_store.execute_command(command_str, cmd_context)

            # Stream output as session updates
            updates = output_writer.get_session_updates()
            for update in updates:
                yield update

        except Exception as e:
            logger.exception("Command execution failed")
            # Send error as session update
            error_updates = to_session_updates(f"Command error: {e}", session.session_id)
            for update in error_updates:
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
        # Get agent context from session
        agent_context = session.agent.context

        # Create command context - cast to satisfy type checker
        command_store: CommandStore = self.command_store
        return command_store.create_context(
            data=agent_context,
            output_writer=output_writer,  # type: ignore
        )

    def get_commands_by_category(
        self, category: str | None = None
    ) -> list[AvailableCommand]:
        """Get commands filtered by category.

        Args:
            category: Category to filter by (None for all)

        Returns:
            List of commands in the category
        """
        commands = []

        for command in self.command_store._commands.values():
            cmd_category = getattr(command, "category", None)

            if category is None or cmd_category == category:
                acp_command = self._convert_command(command)
                if acp_command:
                    commands.append(acp_command)

        return commands

    def update_commands(self, new_commands: list[BaseCommand]) -> None:
        """Update command store with new commands.

        Args:
            new_commands: List of new commands to add
        """
        for command in new_commands:
            self.command_store.register_command(command)

        # Notify sessions about command changes
        self._notify_command_update()

    def register_update_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for command updates.

        Args:
            callback: Function to call when commands are updated
        """
        self._update_callbacks.append(callback)

    def _notify_command_update(self) -> None:
        """Notify all registered callbacks about command updates."""
        for callback in self._update_callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Command update callback failed")
