"""Filesystem tools for ACP agent client-side file operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from acp.schema import ReadTextFileRequest, WriteTextFileRequest
from llmling_agent.log import get_logger
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from llmling_agent_acp.acp_agent import LLMlingACPAgent


logger = get_logger(__name__)


def get_filesystem_tools(agent: LLMlingACPAgent) -> list[Tool]:
    """Get filesystem tools for ACP agent.

    Args:
        agent: The ACP agent instance

    Returns:
        List of filesystem tools
    """
    return [
        Tool.from_callable(
            _create_read_text_file_tool(agent),
            source="filesystem",
            name_override="read_text_file",
        ),
        Tool.from_callable(
            _create_write_text_file_tool(agent),
            source="filesystem",
            name_override="write_text_file",
        ),
    ]


def _create_read_text_file_tool(agent: LLMlingACPAgent):
    """Create a tool that reads text files via the ACP client."""

    async def read_text_file(
        path: str,
        line: int | None = None,
        limit: int | None = None,
        session_id: str = "default_session",
    ) -> str:
        """Read text file contents from the client's filesystem.

        Args:
            path: Absolute path to the file to read
            line: Optional line number to start reading from (1-based)
            limit: Optional maximum number of lines to read
            session_id: Session ID for the request

        Returns:
            File content or error message
        """
        try:
            request = ReadTextFileRequest(
                session_id=session_id,
                path=path,
                line=line,
                limit=limit,
            )
            response = await agent.connection.read_text_file(request)
        except Exception as e:  # noqa: BLE001
            return f"Error reading file: {e}"
        else:
            return response.content

    return read_text_file


def _create_write_text_file_tool(agent: LLMlingACPAgent):
    """Create a tool that writes text files via the ACP client."""

    async def write_text_file(
        path: str,
        content: str,
        session_id: str = "default_session",
    ) -> str:
        """Write text content to a file in the client's filesystem.

        Args:
            path: Absolute path to the file to write
            content: The text content to write to the file
            session_id: Session ID for the request

        Returns:
            Success message or error message
        """
        try:
            request = WriteTextFileRequest(
                session_id=session_id,
                path=path,
                content=content,
            )
            await agent.connection.write_text_file(request)
        except Exception as e:  # noqa: BLE001
            return f"Error writing file: {e}"
        else:
            return f"Successfully wrote file: {path}"

    return write_text_file
