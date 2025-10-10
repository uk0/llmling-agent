"""ACP filesystem provider for file read/write operations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext  # noqa: TC002

from acp.schema import (
    ContentToolCallContent,
    ReadTextFileRequest,
    SessionNotification,
    TextContentBlock,
    ToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
    WriteTextFileRequest,
)
from llmling_agent.log import get_logger
from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from acp.schema import ClientCapabilities
    from llmling_agent_acp.acp_agent import LLMlingACPAgent


logger = get_logger(__name__)


class ACPFileSystemProvider(ResourceProvider):
    """Provides ACP filesystem-related tools for file operations.

    This provider creates session-aware tools for reading and writing files
    via the ACP client. All tools have the session ID baked in at creation time,
    eliminating the need for parameter injection.
    """

    def __init__(
        self,
        agent: LLMlingACPAgent,
        session_id: str,
        client_capabilities: ClientCapabilities,
        cwd: str | None = None,
    ):
        """Initialize filesystem provider.

        Args:
            agent: The ACP agent instance
            session_id: Session ID for all tools created by this provider
            client_capabilities: Client-reported capabilities
            cwd: Current working directory for relative path resolution
        """
        super().__init__(name=f"acp_filesystem_{session_id}")
        self.agent = agent
        self.session_id = session_id
        self.client_capabilities = client_capabilities
        self.cwd = cwd

    def _resolve_path(self, path: str) -> str:
        """Resolve a potentially relative path to an absolute path.

        If cwd is set and path is relative, resolves relative to cwd.
        Otherwise returns the path as-is.

        Args:
            path: Path that may be relative or absolute

        Returns:
            Absolute path string
        """
        if self.cwd and not (path.startswith("/") or (len(path) > 1 and path[1] == ":")):
            # Path appears to be relative and we have a cwd

            return str(Path(self.cwd) / path)
        return path

    async def get_tools(self) -> list[Tool]:
        """Get filesystem tools based on client capabilities."""
        tools: list[Tool] = []

        # Filesystem tools if supported
        if fs_caps := self.client_capabilities.fs:
            # Convert dict to FileSystemCapability if needed
            if isinstance(fs_caps, dict):
                from acp.schema import FileSystemCapability

                fs_caps = FileSystemCapability(
                    read_text_file=fs_caps.get("readTextFile", False),
                    write_text_file=fs_caps.get("writeTextFile", False),
                )

            if fs_caps.read_text_file:
                tool = Tool.from_callable(
                    self._create_read_text_file_tool(),
                    source="filesystem",
                    name_override="read_text_file",
                )
                tools.append(tool)
            if fs_caps.write_text_file:
                tool = Tool.from_callable(
                    self._create_write_text_file_tool(),
                    source="filesystem",
                    name_override="write_text_file",
                )
                tools.append(tool)

        return tools

    def _create_read_text_file_tool(self):
        """Create a tool that reads text files via the ACP client."""

        async def read_text_file(  # noqa: D417
            ctx: RunContext[Any],
            path: str,
            line: int | None = None,
            limit: int | None = None,
        ) -> str:
            r"""Read the contents of a text file.

            Use this to read configuration files, source code,
            logs, or any text-based files from the client's filesystem.

            Args:
                path: File path (absolute or relative to session cwd)
                line: Optional line number to start reading from (1-based indexing)
                limit: Optional maximum number of lines to read from the starting line

            Returns:
                Complete file contents as text, or error message if file cannot be read

            Example:
                read_text_file('src/main.py') -> 'def main():\\n    pass'
            """
            # Resolve relative paths against session cwd
            resolved_path = self._resolve_path(path)

            # Send initial pending notification
            assert ctx.tool_call_id, "Tool call ID must be present for fs operations"
            try:
                start = ToolCallStart(
                    tool_call_id=ctx.tool_call_id,
                    status="pending",
                    title=f"Reading file: {path}",
                    kind="read",
                    locations=[ToolCallLocation(path=resolved_path)],
                )
                notifi = SessionNotification(session_id=self.session_id, update=start)
                await self.agent.connection.session_update(notifi)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to send pending update: %s", e)

            request = ReadTextFileRequest(
                session_id=self.session_id,
                path=resolved_path,
                line=line,
                limit=limit,
            )

            try:
                response = await self.agent.connection.read_text_file(request)

                # Send completion update
                assert ctx.tool_call_id, "Tool call ID must be present for fs operations"
                block = TextContentBlock(text=f"````\n{response.content}\n````")
                progress = ToolCallProgress(
                    tool_call_id=ctx.tool_call_id,
                    status="completed",
                    locations=[ToolCallLocation(path=resolved_path)],
                    content=[ContentToolCallContent(content=block)],
                )
                notifi = SessionNotification(session_id=self.session_id, update=progress)
                try:
                    await self.agent.connection.session_update(notifi)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to send completed update: %s", e)

            except Exception as e:  # noqa: BLE001
                # Send failed update
                assert ctx.tool_call_id, "Tool call ID must be present for fs operations"
                progress = ToolCallProgress(
                    tool_call_id=ctx.tool_call_id,
                    status="failed",
                    raw_output=f"Error: {e}",
                )
                failed = SessionNotification(session_id=self.session_id, update=progress)
                try:
                    await self.agent.connection.session_update(failed)
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to send failed update")

                return f"Error reading file: {e}"
            else:
                return response.content

        return read_text_file

    def _create_write_text_file_tool(self):
        """Create a tool that writes text files via the ACP client."""

        async def write_text_file(ctx: RunContext[Any], path: str, content: str) -> str:  # noqa: D417
            r"""Write text content to a file, creating or overwriting as needed.

            Args:
                path: File path (absolute or relative to session cwd)
                content: Text content to write to the file

            Returns:
                Success message or error description

            Example:
                write_text_file('config.json', '{"debug": true}') ->
                'Successfully wrote file: config.json'
            """
            # Resolve relative paths against session cwd
            resolved_path = self._resolve_path(path)

            # Send initial pending notification
            assert ctx.tool_call_id, "Tool call ID must be present for fs operations"
            try:
                update = ToolCallStart(
                    tool_call_id=ctx.tool_call_id,
                    status="pending",
                    title=f"Writing file: {path}",
                    kind="edit",
                    locations=[ToolCallLocation(path=resolved_path)],
                )
                noti = SessionNotification(session_id=self.session_id, update=update)
                await self.agent.connection.session_update(noti)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to send pending update: %s", e)

            request = WriteTextFileRequest(
                session_id=self.session_id,
                path=resolved_path,
                content=content,
            )

            try:
                await self.agent.connection.write_text_file(request)

                # Send completion update
                assert ctx.tool_call_id, "Tool call ID must be present for fs operations"
                try:
                    block = TextContentBlock(text=content)
                    tool_content = ContentToolCallContent(content=block)
                    progress = ToolCallProgress(
                        tool_call_id=ctx.tool_call_id,
                        status="completed",
                        locations=[ToolCallLocation(path=resolved_path)],
                        content=[tool_content],
                    )
                    s = SessionNotification(session_id=self.session_id, update=progress)
                    await self.agent.connection.session_update(s)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to send completed update: %s", e)

            except Exception as e:  # noqa: BLE001
                # Send failed update
                assert ctx.tool_call_id, "Tool call ID must be present for fs operations"
                try:
                    progress = ToolCallProgress(
                        tool_call_id=ctx.tool_call_id,
                        status="failed",
                        raw_output=f"Error: {e}",
                    )
                    s = SessionNotification(session_id=self.session_id, update=progress)
                    await self.agent.connection.session_update(s)
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to send failed update")

                return f"Error writing file: {e}"
            else:
                return f"Successfully wrote file: {path}"

        return write_text_file
