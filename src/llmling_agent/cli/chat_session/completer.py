from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.completion import Completer, Completion


if TYPE_CHECKING:
    from llmling_agent.chat_session.base import AgentChatSession


class CommandCompleter(Completer):
    """Provides command completion for chat sessions."""

    def __init__(self, session: AgentChatSession) -> None:
        """Initialize completer with chat session."""
        self.session = session

    def get_completions(self, document, complete_event):
        """Get command completions."""
        word = document.get_word_before_cursor()
        text = document.text

        # Only complete commands starting with /
        if not text.startswith("/"):
            return

        # Get all available commands
        commands = self.session._command_store.list_commands()

        # If we're at the start of a command
        if " " not in text:
            # Complete command names
            for cmd_obj in commands:
                if cmd_obj.name.startswith(word.lstrip("/")):
                    yield Completion(
                        f"/{cmd_obj.name}",
                        start_position=-len(word),
                        display_meta=cmd_obj.description,
                    )
        else:
            # We're after a command, try to complete arguments
            cmd_name = text.split()[0][1:]  # Remove the /
            if (cmd := self.session._command_store.get_command(cmd_name)) and (
                cmd.name in ("enable-tool", "disable-tool")
            ):
                # Complete with available tool names
                tool_states = self.session.get_tool_states()
                for name in tool_states:
                    if name.startswith(word):
                        current_state = "enabled" if tool_states[name] else "disabled"
                        text = f"currently {current_state}"
                        pos = -len(word)
                        yield Completion(name, start_position=pos, display_meta=text)
