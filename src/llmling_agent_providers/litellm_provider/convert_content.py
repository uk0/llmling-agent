"""Converters for Content types to LiteLLM formats."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import warnings

from llmling_agent.models import content as own_content


if TYPE_CHECKING:
    from llmling_agent.models.content import Content


def content_to_litellm_format(content: Content) -> dict[str, Any] | None:  # noqa: PLR0911
    """Convert our Content types to LiteLLM-compatible format.

    Args:
        content: One of our Content instances to convert

    Returns:
        Converted LiteLLM-compatible content dict or None if conversion not possible

    Note:
        Some content types may not have an equivalent in LiteLLM.
        In these cases, None is returned and a warning is emitted.
    """
    match content:
        case own_content.ImageURLContent():
            return {
                "type": "image_url",
                "image_url": {"url": content.url, "detail": content.detail or "auto"},
            }

        case own_content.ImageBase64Content():
            data_url = f"data:{content.mime_type};base64,{content.data}"
            return {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                    "detail": content.detail or "auto",
                    "format": content.mime_type,
                },
            }

        case own_content.PDFURLContent():
            # LiteLLM treats PDFs as image_url type
            return {
                "type": "image_url",
                "image_url": {
                    "url": content.url,
                    "detail": content.detail or "auto",
                    "format": "application/pdf",
                },
            }

        case own_content.PDFBase64Content():
            # LiteLLM treats PDFs as image_url type
            data_url = f"data:application/pdf;base64,{content.data}"
            return {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                    "detail": content.detail or "auto",
                    "format": "application/pdf",
                },
            }

        case own_content.AudioURLContent():
            return {
                "type": "input_audio",
                "input_audio": {"url": content.url, "format": content.format or "mp3"},
            }

        case own_content.AudioBase64Content():
            return {
                "type": "input_audio",
                "input_audio": {"data": content.data, "format": content.format or "mp3"},
            }

        case own_content.VideoURLContent():
            msg = "Video content not consistently supported by LiteLLM models"
            warnings.warn(msg, UserWarning, stacklevel=2)
            return {
                "type": "video_url",
                "video_url": {"url": content.url, "format": content.format},
            }

        case _:
            msg = f"Unknown content type: {type(content)}"
            warnings.warn(msg, UserWarning, stacklevel=2)
            return None
