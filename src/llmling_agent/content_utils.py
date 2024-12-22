"""Message handling and conversion utilities."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
    UserPromptPart,
)
from upath import UPath


# Type definitions
type ContentType = Literal["text", "image", "audio", "video"]
type ContentSource = str | bytes | Path | Any


def to_base64(data: bytes) -> str:
    """Convert bytes to base64 string."""
    return base64.b64encode(data).decode()


def create_message(
    contents: list[tuple[ContentType, ContentSource]] | str,
    role: Literal["user", "system"] = "user",
) -> ModelMessage:
    """Create a message from content pairs.

    For multi-modal content, creates a JSON string that models like GPT-4V
    can interpret. For simple text, creates a plain text message.
    """
    # Handle simple text case
    if isinstance(contents, str):
        part = (
            UserPromptPart(content=contents)
            if role == "user"
            else SystemPromptPart(content=contents)
        )
        return ModelRequest(parts=[part])

    # For multi-modal, convert to a JSON string
    content_list = []
    for type_, content in contents:
        match type_:
            case "text":
                content_list.append({"type": "text", "text": str(content)})
            case "image":
                url = prepare_image_url(content)
                content_list.append({"type": "image", "url": url})
            case "audio":
                url = prepare_audio_url(content)
                content_list.append({"type": "audio", "url": url})
            case _:
                msg = f"Unsupported content type: {type_}"
                raise ValueError(msg)

    # Convert to JSON string and create appropriate message part
    content_str = json.dumps({"content": content_list})
    return ModelRequest(
        parts=[
            UserPromptPart(content=content_str)
            if role == "user"
            else SystemPromptPart(content=content_str)
        ]
    )


def prepare_image_url(content: ContentSource) -> str:
    """Convert image content to URL or data URL."""
    match content:
        case str() if content.startswith(("http://", "https://")):
            return content
        case str() | os.PathLike():
            # Read file and convert to data URL
            path = UPath(content)
            content_b64 = to_base64(path.read_bytes())
            return f"data:image/png;base64,{content_b64}"
        case bytes():
            content_b64 = to_base64(content)
            return f"data:image/png;base64,{content_b64}"
        case _:
            msg = f"Unsupported image content type: {type(content)}"
            raise ValueError(msg)


def prepare_audio_url(content: ContentSource) -> str:
    """Convert audio content to URL or data URL.

    Supports common audio formats (mp3, wav, ogg, m4a).
    Uses content-type detection when possible.
    """
    import mimetypes

    def get_audio_mime(path: str | Path) -> str:
        """Get MIME type for audio file."""
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type or not mime_type.startswith("audio/"):
            # Default to mp3 if we can't detect or it's not audio
            return "audio/mpeg"
        return mime_type

    match content:
        case str() if content.startswith(("http://", "https://")):
            return content
        case str() | os.PathLike():
            path = UPath(content)
            content_b64 = to_base64(path.read_bytes())
            mime_type = get_audio_mime(path)
            return f"data:{mime_type};base64,{content_b64}"
        case bytes():
            # For raw bytes, default to mp3 as it's most common
            content_b64 = to_base64(content)
            return f"data:audio/mpeg;base64,{content_b64}"
        case _:
            msg = f"Unsupported audio content type: {type(content)}"
            raise ValueError(msg)
