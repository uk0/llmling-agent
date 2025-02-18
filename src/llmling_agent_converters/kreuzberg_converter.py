"""Kreuzberg document converter."""

from __future__ import annotations

import mimetypes
from typing import TYPE_CHECKING, Any, ClassVar

from kreuzberg import extract_bytes, extract_file

from llmling_agent.log import get_logger
from llmling_agent.utils.tasks import TaskManagerMixin
from llmling_agent_config.converters import KreuzbergConfig
from llmling_agent_converters.base import DocumentConverter


if TYPE_CHECKING:
    from llmling_agent.common_types import StrPath

logger = get_logger(__name__)


class KreuzbergConverter(DocumentConverter, TaskManagerMixin):
    """Converter using Kreuzberg for document conversion.

    Handles:
    - PDFs (searchable and scanned)
    - Office documents
    - Images with OCR
    - Various markup formats
    """

    # All formats supported by Kreuzberg
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {
        # PDFs
        "application/pdf",
        # Office documents
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.oasis.opendocument.text",
        "application/rtf",
        # Ebooks and markup
        "application/epub+zip",
        "text/html",
        "text/markdown",
        "text/plain",
        "text/x-rst",
        "text/org",
        # Images for OCR
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/bmp",
        "image/webp",
        "image/gif",
    }

    def __init__(self, config: KreuzbergConfig | None = None):
        """Initialize converter with config.

        Args:
            config: Optional configuration for Kreuzberg.
                   Uses default settings if not provided.
        """
        super().__init__()
        self.config = config or KreuzbergConfig()

    def supports_file(self, path: StrPath) -> bool:
        """Check if Kreuzberg can handle this file type."""
        mime_type, _ = mimetypes.guess_type(str(path))
        return mime_type in self.SUPPORTED_MIME_TYPES

    def supports_content(self, content: Any, mime_type: str | None = None) -> bool:
        """Check if content type is supported."""
        return mime_type in self.SUPPORTED_MIME_TYPES

    async def _convert_file_async(self, path: StrPath) -> str:
        """Async implementation of file conversion."""
        result = await extract_file(
            path,
            force_ocr=self.config.force_ocr,
            language=self.config.language,
            max_processes=self.config.max_processes,
        )
        return result.content

    def convert_file(self, path: StrPath) -> str:
        """Convert file using Kreuzberg's specialized file API."""
        try:
            return self.run_task_sync(self._convert_file_async(path))
        except Exception as e:
            msg = f"Failed to convert file {path}"
            logger.exception(msg)
            raise ValueError(msg) from e

    async def _convert_content_async(
        self, content: bytes | str, mime_type: str | None = None
    ) -> str:
        """Async implementation of content conversion."""
        # Convert string content to bytes if needed
        if isinstance(content, str):
            content = content.encode()

        result = await extract_bytes(
            content,
            mime_type=mime_type or "text/plain",
            force_ocr=self.config.force_ocr,
            language=self.config.language,
            max_processes=self.config.max_processes,
        )
        return result.content

    def convert_content(self, content: bytes | str, mime_type: str | None = None) -> str:
        """Convert content using Kreuzberg."""
        if not mime_type:
            msg = "MIME type required for content conversion"
            raise ValueError(msg)

        if not self.supports_content(content, mime_type):
            msg = f"Unsupported content type: {mime_type}"
            raise ValueError(msg)

        try:
            return self.run_task_sync(self._convert_content_async(content, mime_type))
        except Exception as e:
            msg = "Failed to convert content"
            logger.exception(msg)
            raise ValueError(msg) from e


if __name__ == "__main__":
    converter = KreuzbergConverter()
