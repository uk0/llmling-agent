"""Base class for converters."""

from __future__ import annotations

from abc import ABC, abstractmethod
import mimetypes
from typing import TYPE_CHECKING, Any

from upath import UPath


if TYPE_CHECKING:
    from llmling_agent.common_types import StrPath


class DocumentConverter(ABC):
    """Base class for document converters."""

    def convert_file(self, path: StrPath) -> str:
        """Convert document file to markdown."""
        path_obj = UPath(path)
        content = path_obj.read_bytes()
        return self.convert_content(content, mimetypes.guess_type(str(path))[0])

    @abstractmethod
    def convert_content(self, content: Any, mime_type: str | None = None) -> str:
        """Convert content to markdown."""

    @abstractmethod
    def supports_file(self, path: StrPath) -> bool:
        """Check if converter can handle this file type."""

    @abstractmethod
    def supports_content(self, content: Any, mime_type: str | None = None) -> bool:
        """Check if converter can handle this content type."""
