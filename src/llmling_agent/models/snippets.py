"""Snippet models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Snippet:
    """Content to be included in the next message."""

    source: str
    content: str

    def format(self) -> str:
        """Format snippet for inclusion in prompt."""
        return f"Content from {self.source}:\n\n{self.content}\n"
