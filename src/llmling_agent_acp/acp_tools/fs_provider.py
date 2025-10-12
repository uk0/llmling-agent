"""ACP filesystem provider for file read/write operations."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext  # noqa: TC002

from acp.schema import (
    ContentToolCallContent,
    FileEditToolCallContent,
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
from llmling_agent_tools.file_editor import replace_content


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
            if fs_caps.read_text_file:
                tool = Tool.from_callable(
                    self.read_text_file, source="filesystem", category="read"
                )
                tools.append(tool)
            if fs_caps.write_text_file:
                tool = Tool.from_callable(
                    self.write_text_file, source="filesystem", category="edit"
                )
                tools.append(tool)

            # Edit file tool requires both read and write capabilities
            if fs_caps.read_text_file and fs_caps.write_text_file:
                tool = Tool.from_callable(
                    self.edit_file, source="filesystem", category="edit"
                )
                tools.append(tool)
                tool = Tool.from_callable(
                    self.agentic_edit, source="filesystem", category="edit"
                )
                tools.append(tool)

        return tools

    async def read_text_file(  # noqa: D417
        self,
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

    async def write_text_file(self, ctx: RunContext[Any], path: str, content: str) -> str:  # noqa: D417
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

    async def edit_file(  # noqa: D417
        self,
        ctx: RunContext[Any],
        path: str,
        old_string: str,
        new_string: str,
        description: str,
        replace_all: bool = False,
    ) -> str:
        r"""Edit a file by replacing specific content with smart matching.

        Uses sophisticated matching strategies to handle whitespace, indentation,
        and other variations. Shows the changes as a diff in the UI.

        Args:
            path: File path (absolute or relative to session cwd)
            old_string: Text content to find and replace
            new_string: Text content to replace it with
            description: Human-readable description of what the edit accomplishes
            replace_all: Whether to replace all occurrences (default: False)

        Returns:
            Success message with edit summary
        """
        if old_string == new_string:
            return "Error: old_string and new_string must be different"

        resolved_path = self._resolve_path(path)  # Resolve paths against session cwd
        # Send initial pending notification
        assert ctx.tool_call_id, "Tool call ID must be present for edit operations"
        start = ToolCallStart(
            tool_call_id=ctx.tool_call_id,
            status="pending",
            title=f"Editing file: {path}",
            kind="edit",
            locations=[ToolCallLocation(path=resolved_path)],
            raw_input={
                "path": path,
                "old_string": old_string,
                "new_string": new_string,
                "description": description,
                "replace_all": replace_all,
            },
        )
        notifi = SessionNotification(session_id=self.session_id, update=start)
        try:
            await self.agent.connection.session_update(notifi)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to send pending update: %s", e)

        try:  # Read current file content
            read_req = ReadTextFileRequest(session_id=self.session_id, path=resolved_path)
            read_response = await self.agent.connection.read_text_file(read_req)
            original_content = read_response.content

            try:  # Apply smart content replacement
                new_content = replace_content(
                    original_content, old_string, new_string, replace_all
                )
            except ValueError as e:
                error_msg = f"Edit failed: {e}"
                # Send failed update
                progress = ToolCallProgress(
                    tool_call_id=ctx.tool_call_id,
                    status="failed",
                    raw_output=error_msg,
                )
                failed = SessionNotification(session_id=self.session_id, update=progress)
                try:
                    await self.agent.connection.session_update(failed)
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to send failed update")
                return error_msg

            # Generate diff for UI display
            diff_lines = list(
                difflib.unified_diff(
                    original_content.splitlines(keepends=True),
                    new_content.splitlines(keepends=True),
                    fromfile=path,
                    tofile=path,
                    lineterm="",
                )
            )

            # Write the new content
            write_request = WriteTextFileRequest(
                session_id=self.session_id,
                path=resolved_path,
                content=new_content,
            )
            await self.agent.connection.write_text_file(write_request)

            # Send completion update with diff
            file_edit_content = FileEditToolCallContent(
                path=resolved_path,
                old_text=original_content,
                new_text=new_content,
            )

            lines_changed = len([
                line for line in diff_lines if line.startswith(("+", "-"))
            ])

            success_msg = f"Successfully edited {Path(path).name}: {description}"
            if lines_changed > 0:
                success_msg += f" ({lines_changed} lines changed)"

            progress = ToolCallProgress(
                tool_call_id=ctx.tool_call_id,
                status="completed",
                locations=[ToolCallLocation(path=resolved_path)],
                content=[file_edit_content],
            )
            notifi = SessionNotification(session_id=self.session_id, update=progress)
            await self.agent.connection.session_update(notifi)
        except Exception as e:  # noqa: BLE001
            error_msg = f"Error editing file: {e}"
            # Send failed update
            progress = ToolCallProgress(
                tool_call_id=ctx.tool_call_id,
                status="failed",
                raw_output=error_msg,
            )
            failed = SessionNotification(session_id=self.session_id, update=progress)
            try:
                await self.agent.connection.session_update(failed)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to send failed update")
            return error_msg
        else:
            return success_msg

    async def agentic_edit(  # noqa: D417, PLR0915
        self,
        ctx: RunContext[Any],
        path: str,
        instructions: str,
    ) -> str:
        r"""Edit a file using AI agent with natural language instructions.

        Creates a new agent that reads the file and rewrites it based on the instructions.
        Shows real-time progress and diffs as the agent works.

        Args:
            path: File path (absolute or relative to session cwd)
            instructions: Natural language instructions for how to modify the file

        Returns:
            Success message with edit summary

        Example:
            agentic_edit('src/main.py', 'Add error handling to the main function') ->
            'Successfully edited main.py using AI agent'
        """
        resolved_path = self._resolve_path(path)

        # Send initial pending notification
        assert ctx.tool_call_id, (
            "Tool call ID must be present for agentic edit operations"
        )
        start = ToolCallStart(
            tool_call_id=ctx.tool_call_id,
            status="pending",
            title=f"AI editing file: {path}",
            kind="edit",
            locations=[ToolCallLocation(path=resolved_path)],
            raw_input={"path": path, "instructions": instructions},
        )
        notifi = SessionNotification(session_id=self.session_id, update=start)
        try:
            await self.agent.connection.session_update(notifi)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to send pending update: %s", e)

        try:
            # Read current file content
            read_req = ReadTextFileRequest(session_id=self.session_id, path=resolved_path)
            read_response = await self.agent.connection.read_text_file(read_req)
            original_content = read_response.content

            from pydantic_ai import Agent as PydanticAgent

            # Append our edit instruction to the existing conversation
            prompt = f"""Please edit the file {path} according to these instructions:

{instructions}

Current file content:
```
{original_content}
```

Output only the new file content, no explanations or markdown formatting."""

            # Create the editor agent using the same model
            sys_prompt = "You are a code editor. Output ONLY the modified file content."
            editor_agent = PydanticAgent(model=ctx.model, system_prompt=sys_prompt)

            # Stream with full message history for caching
            new_content_parts = []

            async with editor_agent.run_stream(
                prompt, message_history=ctx.messages
            ) as response:
                async for chunk in response.stream_text(delta=True):
                    chunk_str = str(chunk)
                    new_content_parts.append(chunk_str)

                    # Build partial content for progress updates
                    partial_content = "".join(new_content_parts)

                    # Send progress update with current diff
                    try:
                        if (
                            len(partial_content.strip()) > 0
                        ):  # Only send if we have meaningful content
                            file_edit_content = FileEditToolCallContent(
                                path=resolved_path,
                                old_text=original_content,
                                new_text=partial_content,
                            )

                            progress = ToolCallProgress(
                                tool_call_id=ctx.tool_call_id,
                                status="in_progress",
                                locations=[ToolCallLocation(path=resolved_path)],
                                content=[file_edit_content],
                            )
                            notifi = SessionNotification(
                                session_id=self.session_id, update=progress
                            )
                            await self.agent.connection.session_update(notifi)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Failed to send progress update: %s", e)

            # Get final content
            new_content = "".join(new_content_parts).strip()

            if not new_content:
                error_msg = "AI agent produced no output"
                progress = ToolCallProgress(
                    tool_call_id=ctx.tool_call_id,
                    status="failed",
                    raw_output=error_msg,
                )
                failed = SessionNotification(session_id=self.session_id, update=progress)
                try:
                    await self.agent.connection.session_update(failed)
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to send failed update")
                return error_msg

            # Write the new content to file
            write_request = WriteTextFileRequest(
                session_id=self.session_id,
                path=resolved_path,
                content=new_content,
            )
            await self.agent.connection.write_text_file(write_request)

            # Send final completion update with complete diff
            file_edit_content = FileEditToolCallContent(
                path=resolved_path,
                old_text=original_content,
                new_text=new_content,
            )

            # Calculate some stats
            original_lines = len(original_content.splitlines())
            new_lines = len(new_content.splitlines())

            success_msg = f"Successfully edited {Path(path).name} using AI agent"
            success_msg += f" ({original_lines} → {new_lines} lines)"

            progress = ToolCallProgress(
                tool_call_id=ctx.tool_call_id,
                status="completed",
                locations=[ToolCallLocation(path=resolved_path)],
                content=[file_edit_content],
            )
            notifi = SessionNotification(session_id=self.session_id, update=progress)
            await self.agent.connection.session_update(notifi)

        except Exception as e:  # noqa: BLE001
            error_msg = f"Error during agentic edit: {e}"
            # Send failed update
            progress = ToolCallProgress(
                tool_call_id=ctx.tool_call_id,
                status="failed",
                raw_output=error_msg,
            )
            failed = SessionNotification(session_id=self.session_id, update=progress)
            try:
                await self.agent.connection.session_update(failed)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to send failed update")
            return error_msg
        else:
            return success_msg
