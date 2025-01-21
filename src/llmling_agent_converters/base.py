from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
import mimetypes
from typing import TYPE_CHECKING, Any

from upath import UPath


if TYPE_CHECKING:
    from os import PathLike


class DocumentConverter(ABC):
    """Base class for document converters."""

    @abstractproperty
    def is_async(self) -> bool:
        """Whether this converter has native async support."""
        ...

    @abstractmethod
    async def supports_file(self, path: str | PathLike[str]) -> bool:
        """Check if converter can handle this file type."""

    @abstractmethod
    async def supports_content(self, content: Any, mime_type: str | None = None) -> bool:
        """Check if converter can handle this content type."""

    # For sync converters, these are normal methods
    def convert_file(self, path: str | PathLike[str]) -> str:
        """Convert document file to markdown.

        Default implementation reads file and forwards to convert_content.
        Override if converter has special file handling needs.
        """
        path_obj = UPath(path)
        content = path_obj.read_bytes()
        return self.convert_content(content, mimetypes.guess_type(str(path))[0])

    @abstractmethod
    def convert_content(self, content: Any, mime_type: str | None = None) -> str:
        """Convert content to markdown."""

    # For async converters, they override these instead
    async def convert_file_async(self, path: str | PathLike[str]) -> str:
        """Async version of convert_file."""
        raise NotImplementedError

    async def convert_content_async(
        self, content: Any, mime_type: str | None = None
    ) -> str:
        """Async version of convert_content."""
        raise NotImplementedError
