"""Command completion system."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prompt_toolkit.completion import Completer, Completion

from llmling_agent.commands.completion import CompletionContext


if TYPE_CHECKING:
    from collections.abc import Iterator

    from prompt_toolkit.document import Document

    from llmling_agent.commands.base import BaseCommand


class PromptToolkitCompleter(Completer):
    """Adapts our completion system to prompt-toolkit."""

    def __init__(self, commands: dict[str, BaseCommand]) -> None:
        self._commands = commands

    def get_completions(
        self,
        document: Document,
        complete_event: Any,
    ) -> Iterator[Completion]:
        """Get completions for the current context."""
        word_before_cursor = document.get_word_before_cursor()

        # Command completion
        if document.text.startswith("/"):
            word = word_before_cursor.lstrip("/")
            for name, cmd in self._commands.items():
                if name.startswith(word):
                    # Calculate correct start position
                    # If we have "/co", word_before_cursor is "co"
                    # We want to complete from the "c"
                    yield Completion(
                        name,  # completion text
                        start_position=-len(word),  # replace from start of word
                        display_meta=cmd.description,  # show description in menu
                    )
            return

        # If we have a command and it has a completer, use that
        if " " in document.text and document.text.startswith("/"):
            cmd_name = document.text.split()[0][1:]  # remove slash
            command = self._commands.get(cmd_name)
            if command and (completer := command.get_completer()):
                for item in completer.get_completions(CompletionContext(document)):
                    yield item.to_prompt_toolkit(-len(word_before_cursor))
