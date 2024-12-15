"""Command completion system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from collections.abc import Iterator

    from prompt_toolkit.completion import Completion
    from prompt_toolkit.document import Document

    from llmling_agent.commands.base import BaseCommand


@dataclass
class CompletionItem:
    """Single completion suggestion."""

    text: str
    """Text to insert"""

    display: str | None = None
    """Optional display text (defaults to text)"""

    metadata: str | None = None
    """Additional information to show"""

    kind: Literal["command", "file", "tool", "path", "env"] | None = None
    """Type of completion item"""

    sort_text: str | None = None
    """Optional text to use for sorting (defaults to text)"""

    def to_prompt_toolkit(self, start_position: int) -> Completion:
        """Convert to prompt_toolkit completion."""
        from prompt_toolkit.completion import Completion

        return Completion(
            text=self.text,
            start_position=start_position,
            display=self.display or self.text,
            display_meta=self.metadata,
        )


class CompletionContext:
    """Context for completion operations."""

    def __init__(self, document: Document) -> None:
        """Initialize completion context from document."""
        self.document = document
        self._parse_document()

    def _parse_document(self) -> None:
        """Parse document into command and arguments."""
        text = self.document.text.lstrip()

        if not text.startswith("/"):
            self._command_name = None
            self._args = []
            self._current_word = ""
            self._arg_position = 0
            return

        parts = text[1:].split()  # Remove leading slash
        self._command_name = parts[0] if parts else ""
        self._args = parts[1:]
        self._current_word = self.document.get_word_before_cursor()

        # Calculate argument position
        cursor_position = self.document.cursor_position
        text_before_cursor = text[:cursor_position]
        self._arg_position = len(text_before_cursor.split()) - 1

    @property
    def command_name(self) -> str | None:
        """Get current command name if any."""
        return self._command_name

    @property
    def current_word(self) -> str:
        """Get word being completed."""
        return self._current_word

    @property
    def arg_position(self) -> int:
        """Get current argument position."""
        return self._arg_position

    @property
    def command_args(self) -> list[str]:
        """Get current command arguments."""
        return self._args


class CompletionProvider(ABC):
    """Base class for completion providers."""

    @abstractmethod
    def get_completions(
        self,
        context: CompletionContext,
    ) -> Iterator[CompletionItem]:
        """Get completion suggestions.

        Args:
            context: Current completion context

        Returns:
            Iterator of completion suggestions
        """
        ...


class PathCompletionProvider(CompletionProvider):
    """Provides filesystem path completions."""

    def __init__(
        self,
        file_patterns: list[str] | None = None,
        show_hidden: bool = False,
    ) -> None:
        """Initialize path completion provider.

        Args:
            file_patterns: Optional glob patterns to filter files
            show_hidden: Whether to show hidden files/directories
        """
        self.file_patterns = file_patterns
        self.show_hidden = show_hidden

    def get_completions(
        self,
        context: CompletionContext,
    ) -> Iterator[CompletionItem]:
        """Get path completions."""
        # Get base path from current word
        path = context.current_word or "."
        try:
            base = Path(path).expanduser()
            if not base.exists():
                base = base.parent

            # List directory contents
            for entry in base.iterdir():
                # Skip hidden files unless enabled
                if not self.show_hidden and entry.name.startswith("."):
                    continue

                # Check file patterns
                if (
                    self.file_patterns
                    and entry.is_file()
                    and not any(entry.match(p) for p in self.file_patterns)
                ):
                    continue

                # Create completion item
                name = str(entry)
                if entry.is_dir():
                    name = f"{name}{os.sep}"
                    kind = "path"
                else:
                    kind = "file"
                meta = "directory" if entry.is_dir() else None
                yield CompletionItem(text=name, kind=kind, metadata=meta)  # type: ignore[arg-type]

        except Exception:  # noqa: BLE001
            # Fail silently for path completion
            pass


class CommandCompleter:
    """Main command completion implementation."""

    def __init__(self, commands: dict[str, BaseCommand]) -> None:
        """Initialize command completer.

        Args:
            commands: Mapping of command names to command instances
        """
        self._commands = commands
        self._global_providers: list[CompletionProvider] = []

    def add_global_provider(self, provider: CompletionProvider) -> None:
        """Add a global completion provider."""
        self._global_providers.append(provider)

    def get_completions(
        self,
        document: Document,
        complete_event: Any,
    ) -> Iterator[Completion]:
        """Get completions for the current context.

        This is the main entry point used by prompt_toolkit.
        """
        # Create context
        context = CompletionContext(document)

        # Get current word and position
        word = context.current_word
        word_start_position = -len(word)

        # If at start of command, complete command names
        if not context.command_name:
            for name, cmd in self._commands.items():
                if not name.startswith(word.lstrip("/")):
                    continue
                text = f"/{name}"
                item = CompletionItem(text=text, metadata=cmd.description, kind="command")
                yield item.to_prompt_toolkit(word_start_position)
            return

        # Get command-specific completions
        command = self._commands.get(context.command_name)
        if command and (completer := command.get_completer()):
            for item in completer.get_completions(context):
                yield item.to_prompt_toolkit(word_start_position)

        # Get global completions
        for provider in self._global_providers:
            for item in provider.get_completions(context):
                yield item.to_prompt_toolkit(word_start_position)
