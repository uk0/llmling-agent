"""The main Agent. Can do all sort of crazy things."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from toprompt import AnyPromptType, to_prompt

from llmling_agent.models.content import (
    BaseContent,
    BaseImageContent,
    BasePDFContent,
    Content,
    ImageBase64Content,
)
from llmling_agent.utils.async_read import read_path


if TYPE_CHECKING:
    from collections.abc import Sequence

    import PIL.Image


async def convert_prompts(
    prompts: Sequence[AnyPromptType | PIL.Image.Image | os.PathLike[str] | Content],
) -> list[str | Content]:
    """Convert prompts to our internal format.

    Handles:
    - PIL Images -> ImageBase64Content
    - UPath/PathLike -> Auto-detect and convert to appropriate Content
    - Regular prompts -> str via to_prompt
    - Content objects -> pass through
    """
    import PIL.Image
    from upath import UPath

    result: list[str | Content] = []
    for p in prompts:
        match p:
            case PIL.Image.Image():
                result.append(ImageBase64Content.from_pil_image(p))

            case os.PathLike():
                from mimetypes import guess_type

                path_obj = UPath(p)
                mime_type, _ = guess_type(str(path_obj))

                match mime_type:
                    case "application/pdf":
                        content: Content = await BasePDFContent.from_path(path_obj)
                        result.append(content)
                    case str() if mime_type.startswith("image/"):
                        content = await BaseImageContent.from_path(path_obj)
                        result.append(content)
                    case _:
                        # Non-media or unknown type
                        text = await read_path(path_obj)
                        result.append(text)

            case _ if not isinstance(p, BaseContent):
                result.append(await to_prompt(p))
            case _:
                result.append(p)  # type: ignore
    return result


async def format_prompts(prompts: Sequence[str | Content]) -> str:
    """Format prompts for human readability using to_prompt."""
    from toprompt import to_prompt

    parts = [await to_prompt(p) for p in prompts]
    return "\n\n".join(parts)
