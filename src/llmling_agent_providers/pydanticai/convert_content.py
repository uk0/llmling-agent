"""Converters for Content types to pydantic-ai formats."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING
import warnings

from pydantic_ai.messages import AudioUrl, BinaryContent, DocumentUrl, ImageUrl

from llmling_agent.models import content as own_content


if TYPE_CHECKING:
    from pydantic_ai.messages import UserContent

    from llmling_agent.models.content import Content


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
            binary_data = base64.b64decode(content.data)
            typ = content.mime_type or "image/jpeg"
            return BinaryContent(data=binary_data, media_type=typ)
        case own_content.PDFURLContent():
            return DocumentUrl(url=content.url)
        case own_content.PDFBase64Content():
            binary_data = base64.b64decode(content.data)
            return BinaryContent(binary_data, media_type="application/pdf")
        case own_content.AudioURLContent():
            return AudioUrl(url=content.url)
        case own_content.AudioBase64Content():
            binary_data = base64.b64decode(content.data)
            typ = f"audio/{content.format or 'mp3'}"
            return BinaryContent(data=binary_data, media_type=typ)
        case own_content.VideoURLContent():
            msg = "Video content not supported by pydantic-ai"
            warnings.warn(msg, UserWarning, stacklevel=2)
            return None
        case _:
            msg = f"Unknown content type: {type(content)}"
            warnings.warn(msg, UserWarning, stacklevel=2)
            return None
