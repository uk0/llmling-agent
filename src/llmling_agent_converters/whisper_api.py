"""YouTube transcript converter."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, ClassVar

from llmling_agent.log import get_logger
from llmling_agent_converters.base import DocumentConverter


if TYPE_CHECKING:
    from llmling_agent.common_types import StrPath
    from llmling_agent.models.converters import WhisperAPIConfig

logger = get_logger(__name__)


class WhisperAPIConverter(DocumentConverter):
    """Converter using OpenAI's Whisper API."""

    SUPPORTED_MIME_TYPES: ClassVar = {
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/ogg",
        "audio/flac",
    }

    def __init__(self, config: WhisperAPIConfig):
        """Initialize converter with config."""
        self.config = config

    def supports_file(self, path: StrPath) -> bool:
        """Check if file type is supported."""
        import mimetypes

        mime_type, _ = mimetypes.guess_type(str(path))
        return mime_type in self.SUPPORTED_MIME_TYPES

    def supports_content(self, content: Any, mime_type: str | None = None) -> bool:
        """Check if content type is supported."""
        return mime_type in self.SUPPORTED_MIME_TYPES

    def convert_content(self, content: bytes, mime_type: str | None = None) -> str:
        """Convert audio content to text."""
        if not self.supports_content(content, mime_type):
            msg = f"Unsupported audio format: {mime_type}"
            raise ValueError(msg)

        import openai

        openai.api_key = self.config.api_key
        response = openai.Audio.transcribe(
            "whisper-1",
            io.BytesIO(content),
            language=self.config.language,
        )
        return response["text"]
