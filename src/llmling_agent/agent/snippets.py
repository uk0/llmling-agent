from collections.abc import Sequence

from llmling_agent.models.snippets import Snippet


class SnippetManager:
    """Manages pending snippets for inclusion in messages."""

    def __init__(self):
        self._pending_snippets: list[Snippet] = []

    def add(self, snippet: Snippet):
        """Add a snippet to be included in next message."""
        self._pending_snippets.append(snippet)

    def add_snippet(self, content: str, source: str) -> Snippet:
        snippet = Snippet(content=content, source=source)
        self.add(snippet)
        return snippet

    def clear(self):
        """Clear all pending snippets."""
        self._pending_snippets.clear()

    def get_all(self) -> Sequence[Snippet]:
        """Get all pending snippets (read-only)."""
        return list(self._pending_snippets)

    def format_all(self) -> str | None:
        """Format all pending snippets into a single text.

        Returns None if no snippets are pending.
        """
        if not self._pending_snippets:
            return None
        return "\n".join(s.format() for s in self._pending_snippets)

    def has_pending(self) -> bool:
        """Check if there are any pending snippets."""
        return bool(self._pending_snippets)
