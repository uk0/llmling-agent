"""Plain text (noop) converter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from upath import UPath

from llmling_agent.log import get_logger
from llmling_agent_config.converters import PlainConverterConfig
from llmling_agent_converters.base import DocumentConverter


if TYPE_CHECKING:
    from llmling_agent.common_types import StrPath


logger = get_logger(__name__)


class PlainConverter(DocumentConverter):
    """Fallback converter that handles plain text."""

    def __init__(self, config: PlainConverterConfig | None = None):
        self.config = config or PlainConverterConfig()

    def supports_file(self, path: StrPath) -> bool:
        """Support text files or unknown types as last resort."""
        import mimetypes

        mime_type, _ = mimetypes.guess_type(str(path))
        return mime_type is None or mime_type.startswith("text/")

    def supports_content(self, content: Any, mime_type: str | None = None) -> bool:
        """Accept any content we can convert to string."""
        return True

    def convert_file(self, path: StrPath) -> str:
        """Just read the file as text."""
        try:
            return UPath(path).read_text()
        except Exception as e:
            msg = f"Failed to read file {path}"
            logger.exception(msg)
            raise ValueError(msg) from e

    def convert_content(self, content: Any, mime_type: str | None = None) -> str:
        """Return content as string."""
        return str(content)
