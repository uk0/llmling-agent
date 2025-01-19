"""Content types for messages."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from upath import UPath


if TYPE_CHECKING:
    import PIL.Image

    from llmling_agent.common_types import StrPath


DetailLevel = Literal["high", "low", "auto"]


class BaseContent(BaseModel):
    """Base class for special content types (non-text)."""

    type: str = Field(init=False)
    """Discriminator field for content types."""

    description: str | None = None
    """Human-readable description of the content."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class BaseImageContent(BaseContent):
    """Base for image content."""

    detail: DetailLevel | None = None
    """Detail level for image processing by vision models.
    - high: Maximum resolution (up to 2048x2048)
    - low: Lower resolution (512x512)
    - auto: Let model decide based on content
    """

    @classmethod
    async def from_path(
        cls,
        path: StrPath,
        *,
        detail: DetailLevel | None = None,
        description: str | None = None,
    ) -> ImageURLContent | ImageBase64Content:
        """Create image content from any path.

        Automatically chooses between URL and base64 based on path type.
        Downloads and converts remote content if needed.

        Args:
            path: Local path or URL to image
            detail: Optional detail level for processing
            description: Optional description of the image
        """
        path_obj = UPath(path)

        # For http(s) URLs, pass through as URL content
        if path_obj.protocol in ("http", "https"):
            return ImageURLContent(
                url=str(path_obj), detail=detail, description=description
            )

        # For all other paths, read and convert to base64
        content = base64.b64encode(path_obj.read_bytes()).decode()
        return ImageBase64Content(data=content, detail=detail, description=description)


class ImageURLContent(BaseImageContent):
    """Image from URL."""

    type: Literal["image_url"] = Field("image_url", init=False)
    """Type discriminator for URL-based images."""

    url: str
    """URL to the image."""


class ImageBase64Content(BaseImageContent):
    """Image from base64 data."""

    type: Literal["image_base64"] = Field("image_base64", init=False)
    """Type discriminator for base64-encoded images."""

    data: str
    """Base64-encoded image data."""

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        detail: DetailLevel | None = None,
        description: str | None = None,
    ) -> ImageBase64Content:
        """Create image content from raw bytes.

        Args:
            data: Raw image bytes
            detail: Optional detail level for processing
            description: Optional description of the image
        """
        content = base64.b64encode(data).decode()
        return cls(data=content, detail=detail, description=description)

    @classmethod
    def from_pil_image(cls, image: PIL.Image.Image) -> ImageBase64Content:
        """Create content from PIL Image."""
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return cls(data=base64.b64encode(buffer.getvalue()).decode())


Content = Annotated[ImageURLContent | ImageBase64Content, Field(discriminator="type")]
