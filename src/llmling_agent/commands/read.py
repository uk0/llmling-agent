"""Command for reading file content into conversations."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

from slashed import Command, CommandContext, CommandError
from upath import UPath

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.chat_session.base import AgentChatSession


logger = get_logger(__name__)


READ_HELP = """\
Read content from files or URLs into the conversation.

By default reads raw content, but can convert supported formats to markdown
with the --convert-to-md flag.

Supported formats for conversion:
- PDF documents
- Office files (Word, Excel, PowerPoint)
- Images (with metadata)
- Audio files (metadata)
- HTML pages
- Text formats (CSV, JSON, XML)

Examples:
  /read document.txt               # Read raw text
  /read document.pdf --convert-to-md   # Convert PDF to markdown
  /read https://example.com/doc.docx --convert-to-md
  /read presentation.pptx --convert-to-md
"""


async def read_command(
    ctx: CommandContext[AgentChatSession],
    args: list[str],
    kwargs: dict[str, str],
):
    """Read file content into conversation.

    Args:
        ctx: Command context
        args: Command arguments (path)
        kwargs: Command options
    """
    if not args:
        await ctx.output.print("Usage: /read <path_or_url> [--convert-to-md]")
        return

    path = UPath(args[0])
    convert_to_md = kwargs.get("convert_to_md", "").lower() in ("true", "1", "yes")

    try:
        # Handle URLs by downloading first
        if str(path).startswith(("http://", "https://")):
            await ctx.output.print(f"Downloading {path}...")
            bytes_content = path.read_bytes()

            # Create temp file in system temp directory
            temp_dir = UPath(tempfile.gettempdir())
            temp_path = temp_dir / f"llmling_{path.name}"
            temp_path.write_bytes(bytes_content)
            path = temp_path

        # Convert content if requested
        if convert_to_md:
            try:
                await ctx.output.print(f"Converting {path} to markdown...")
                from markitdown import MarkItDown

                md = MarkItDown()
                result = md.convert(str(path))
                content = result.text_content
            except Exception as e:
                msg = f"Failed to convert {path}: {e}"
                raise CommandError(msg) from e
        else:
            # Read raw content
            try:
                content = path.read_text()
            except UnicodeDecodeError:
                msg = f"Unable to read {path} as text. Try --convert-to-md for binaries."
                raise CommandError(msg)  # noqa: B904

        # Send content as user message
        await ctx.data.send_message(content)

    except Exception as e:
        msg = f"Error reading {path}: {e}"
        logger.exception(msg)
        raise CommandError(msg) from e


read_cmd = Command(
    name="read",
    description="Read file or URL content into conversation",
    execute_func=read_command,
    usage="<path_or_url> [--convert-to-md]",
    help_text=READ_HELP,
    category="content",
)
