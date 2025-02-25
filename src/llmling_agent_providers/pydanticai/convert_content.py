"""Converters for Content types to pydantic-ai formats."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

from pydantic_ai.messages import BinaryContent, ImageUrl, UserContent

from llmling_agent_config import content as own_content


if TYPE_CHECKING:
    from llmling_agent_config.content import Content


def content_to_pydantic_ai(content: Content) -> UserContent | None:  # noqa: PLR0911
    """Convert our Content types to pydantic-ai content types.

    Args:
        content: One of our Content instances to convert

    Returns:
        Converted pydantic-ai content or None if conversion not possible

    Note:
        Some content types may not have an equivalent in pydantic-ai.
        In these cases, None is returned and a warning is emitted.
    """
    match content:
        case own_content.ImageURLContent():
            return ImageUrl(url=content.url)

        case own_content.ImageBase64Content():
            return BinaryContent(
                data=content.data.encode(),
                media_type="image/jpeg",
            )

        case own_content.PDFURLContent() | own_content.PDFBase64Content():
            msg = "PDF content not supported by pydantic-ai"
            warnings.warn(msg, UserWarning, stacklevel=2)
            return None

        case own_content.AudioURLContent():
            msg = "Audio URL content not yet implemented"
            warnings.warn(msg, UserWarning, stacklevel=2)
            return None

        case own_content.AudioBase64Content():
            return BinaryContent(
                data=content.data.encode(),
                media_type=f"audio/{content.format or 'mp3'}",
            )

        case own_content.VideoURLContent():
            msg = "Video content not supported by pydantic-ai"
            warnings.warn(msg, UserWarning, stacklevel=2)
            return None

        case _:
            msg = f"Unknown content type: {type(content)}"
            warnings.warn(msg, UserWarning, stacklevel=2)
            return None
