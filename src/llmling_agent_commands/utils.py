"""Command utilities."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING
import webbrowser

from slashed import CommandContext, CommandError, SlashedCommand


if TYPE_CHECKING:
    from llmling_agent.chat_session.base import AgentPoolView


class CopyClipboardCommand(SlashedCommand):
    """Copy messages from conversation history to system clipboard.

    Allows copying a configurable number of messages with options for:
    - Number of messages to include
    - Including/excluding system messages
    - Token limit for context size
    - Custom format templates

    Requires pyperclip package to be installed.
    """

    name = "copy-clipboard"
    category = "utils"

    async def execute_command(
        self,
        ctx: CommandContext[AgentPoolView],
        *,
        num_messages: int = 1,
        include_system: bool = False,
        max_tokens: int | None = None,
        format_template: str | None = None,
    ):
        """Copy messages to clipboard.

        Args:
            ctx: Command context
            num_messages: Number of messages to copy (default: 1)
            include_system: Include system messages
            max_tokens: Only include messages up to token limit
            format_template: Custom format template
        """
        try:
            import pyperclip
        except ImportError as e:
            msg = "pyperclip package required for clipboard operations"
            raise CommandError(msg) from e

        content = await ctx.get_data()._agent.conversation.format_history(
            num_messages=num_messages,
            include_system=include_system,
            max_tokens=max_tokens,
            format_template=format_template,
        )

        if not content.strip():
            await ctx.output.print("No messages found to copy")
            return

        try:
            pyperclip.copy(content)
            await ctx.output.print("Messages copied to clipboard")
        except Exception as e:
            msg = f"Failed to copy to clipboard: {e}"
            raise CommandError(msg) from e

    @classmethod
    def condition(cls) -> bool:
        """Check if pyperclip is available."""
        return importlib.util.find_spec("pyperclip") is not None


class EditAgentFileCommand(SlashedCommand):
    """Open the agent's configuration file in your default editor.

    This file contains:
    - Agent settings and capabilities
    - System promptss
    - Model configuration
    - Environment references
    - Role definitions

    Note: Changes to the configuration file require reloading the agent.
    """

    name = "open-agent-file"
    category = "utils"

    async def execute_command(self, ctx: CommandContext[AgentPoolView]):
        """Open agent's configuration file."""
        agent = ctx.get_data()._agent
        if not agent.context:
            msg = "No agent context available"
            raise CommandError(msg)

        config = agent.context.config
        if not config.config_file_path:
            msg = "No configuration file path available"
            raise CommandError(msg)

        try:
            webbrowser.open(config.config_file_path)
            await ctx.output.print(
                f"Opening agent configuration: {config.config_file_path}"
            )
        except Exception as e:
            msg = f"Failed to open configuration file: {e}"
            raise CommandError(msg) from e
