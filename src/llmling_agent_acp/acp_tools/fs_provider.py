"""ACP filesystem provider for file read/write operations."""

from __future__ import annotations

import difflib
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from pydantic_ai import ModelRetry, RunContext  # noqa: TC002

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
    from llmling_agent_acp.session import ACPSession


logger = get_logger(__name__)


class ACPFileSystemProvider(ResourceProvider):
    """Provides ACP filesystem-related tools for file operations.

    This provider creates session-aware tools for reading and writing files
    via the ACP client. All tools have the session ID baked in at creation time,
    eliminating the need for parameter injection.
    """

    def __init__(
        self,
        session: ACPSession,
        client_capabilities: ClientCapabilities,
        cwd: str | None = None,
    ):
        """Initialize filesystem provider.

        Args:
            session: Session for all tools created by this provider
            client_capabilities: Client-reported capabilities
            cwd: Current working directory for relative path resolution
        """
        super().__init__(name=f"acp_filesystem_{session.session_id}")
        self.agent = session.acp_agent
        self.session_id = session.session_id
        self.session = session
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
            return str(Path(self.cwd) / path)
        return path

    async def get_tools(self) -> list[Tool]:
        """Get filesystem tools based on client capabilities."""
        tools: list[Tool] = []
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
                progress = ToolCallProgress(  # Send failed update
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

            # Write the new content
            write_request = WriteTextFileRequest(
                session_id=self.session_id,
                path=resolved_path,
                content=new_content,
            )
            await self.agent.connection.write_text_file(write_request)
            file_edit_content = FileEditToolCallContent(
                path=resolved_path,
                old_text=original_content,
                new_text=new_content,
            )

            success_msg = f"Successfully edited {Path(path).name}: {description}"
            diff_lines = get_changed_lines(original_content, new_content, resolved_path)
            if lines_changed := len(diff_lines) > 0:
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
            progress = ToolCallProgress(  # Send failed update
                tool_call_id=ctx.tool_call_id, status="failed", raw_output=error_msg
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
        display_description: str,
        mode: str = "edit",
    ) -> str:
        r"""Edit a file using AI agent with natural language instructions.

        Creates a new agent that processes the file based on the instructions.
        Shows real-time progress and diffs as the agent works.

        Args:
            path: File path (absolute or relative to session cwd)
            display_description: Natural language description of the edits to make
            mode: Edit mode - 'edit', 'create', or 'overwrite' (default: 'edit')

        Returns:
            Success message with edit summary

        Example:
            agentic_edit('src/main.py', 'Add error handling to the main function') ->
            'Successfully edited main.py using AI agent'
        """
        resolved_path = self._resolve_path(path)

        # Send initial pending notification
        assert ctx.tool_call_id, "Tool call ID must be present for edit operations"
        start = ToolCallStart(
            tool_call_id=ctx.tool_call_id,
            status="pending",
            title=f"AI editing file: {path}",
            kind="edit",
            locations=[ToolCallLocation(path=resolved_path)],
            raw_input={
                "path": path,
                "display_description": display_description,
                "mode": mode,
            },
        )
        notifi = SessionNotification(session_id=self.session_id, update=start)
        try:
            await self.agent.connection.session_update(notifi)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to send pending update: %s", e)

        try:
            from pydantic_ai import Agent as PydanticAgent

            # Handle different modes
            if mode == "create":
                # For create mode, don't read existing file
                original_content = ""
                prompt = self._build_create_prompt(path, display_description)
                sys_prompt = (
                    "You are a code generator. Create the requested file content."
                )
            elif mode == "overwrite":
                # For overwrite mode, don't read file - agent
                # already read it via system prompt requirement
                original_content = ""  # Will be set later for diff purposes
                prompt = self._build_overwrite_prompt(path, display_description)
                sys_prompt = (
                    "You are a code editor. Output ONLY the complete new file content."
                )
            else:
                # For edit mode, use structured editing approach
                read_req = ReadTextFileRequest(
                    session_id=self.session_id, path=resolved_path
                )
                read_response = await self.agent.connection.read_text_file(read_req)
                original_content = read_response.content
                prompt = self._build_edit_prompt(path, display_description)
                sys_prompt = (
                    "You are a code editor. Output ONLY structured edits "
                    "using the specified format."
                )

            # Create the editor agent using the same model
            editor_agent = PydanticAgent(model=ctx.model, system_prompt=sys_prompt)

            if mode == "edit":
                # For structured editing, get the full response and parse the edits
                response = await editor_agent.run(prompt, message_history=ctx.messages)
                new_content = await self._apply_structured_edits(
                    original_content, response.output
                )
            else:
                # For overwrite mode we need to read the current content for diff purposes
                if mode == "overwrite":
                    read_req = ReadTextFileRequest(
                        session_id=self.session_id, path=resolved_path
                    )
                    read_response = await self.agent.connection.read_text_file(read_req)
                    original_content = read_response.content
                # For create/overwrite modes, stream the complete content
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
                            if len(partial_content.strip()) > 0:
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
            original_lines = len(original_content.splitlines()) if original_content else 0
            new_lines = len(new_content.splitlines())

            if mode == "create":
                path = Path(path).name
                success_msg = f"Successfully created {path} ({new_lines} lines)"
            else:
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

    def _build_create_prompt(self, path: str, description: str) -> str:
        """Build prompt for create mode."""
        return f"""Create a new file at {path} according to this description:

{description}

Output only the complete file content, no explanations or markdown formatting."""

    def _build_overwrite_prompt(self, path: str, description: str) -> str:
        """Build prompt for overwrite mode."""
        return f"""Rewrite the file {path} according to this description:

{description}

Output only the complete new file content, no explanations or markdown formatting."""

    def _build_edit_prompt(self, path: str, description: str) -> str:
        """Build prompt for structured edit mode."""
        return f"""\
You MUST respond with a series of edits to a file, using the following format:

```
<edits>

<old_text line=10>
OLD TEXT 1 HERE
</old_text>
<new_text>
NEW TEXT 1 HERE
</new_text>

<old_text line=456>
OLD TEXT 2 HERE
</old_text>
<new_text>
NEW TEXT 2 HERE
</new_text>

</edits>
```

# File Editing Instructions

- Use `<old_text>` and `<new_text>` tags to replace content
- `<old_text>` must exactly match existing file content, including indentation
- `<old_text>` must come from the actual file, not an outline
- `<old_text>` cannot be empty
- `line` should be a starting line number for the text to be replaced
- Be minimal with replacements:
  - For unique lines, include only those lines
  - For non-unique lines, include enough context to identify them
- Do not escape quotes, newlines, or other characters within tags
- For multiple occurrences, repeat the same tag pair for each instance
- Edits are sequential - each assumes previous edits are already applied
- Only edit the specified file
- Always close all tags properly

<file_to_edit>
{path}
</file_to_edit>

<edit_description>
{description}
</edit_description>

Tool calls have been disabled. You MUST start your response with <edits>."""

    async def _apply_structured_edits(
        self, original_content: str, edits_response: str
    ) -> str:
        """Apply structured edits from the agent response."""
        # Parse the edits from the response
        edits_match = re.search(r"<edits>(.*?)</edits>", edits_response, re.DOTALL)
        if not edits_match:
            logger.warning("No edits block found in response")
            return original_content

        edits_content = edits_match.group(1)

        # Find all old_text/new_text pairs
        old_text_pattern = r"<old_text[^>]*>(.*?)</old_text>"
        new_text_pattern = r"<new_text>(.*?)</new_text>"

        old_texts = re.findall(old_text_pattern, edits_content, re.DOTALL)
        new_texts = re.findall(new_text_pattern, edits_content, re.DOTALL)

        if len(old_texts) != len(new_texts):
            logger.warning("Mismatch between old_text and new_text blocks")
            return original_content

        # Apply edits sequentially
        content = original_content
        applied_edits = 0

        failed_matches = []
        multiple_matches = []

        for old_text, new_text in zip(old_texts, new_texts, strict=False):
            old_text = old_text.strip()
            new_text = new_text.strip()

            # Check for multiple matches (ambiguity)
            match_count = content.count(old_text)
            if match_count > 1:
                multiple_matches.append(old_text[:50])
            elif match_count == 1:
                content = content.replace(old_text, new_text, 1)
                applied_edits += 1
            else:
                failed_matches.append(old_text[:50])

        # Raise ModelRetry for specific failure cases
        if applied_edits == 0 and len(old_texts) > 0:
            msg = (
                "Some edits were produced but none of them could be applied. "
                "Read the relevant sections of the file again so that "
                "I can perform the requested edits."
            )
            raise ModelRetry(msg)

        if multiple_matches:
            msg = (
                f"<old_text> matches multiple positions in the file: {', '.join(multiple_matches)}... "
                "Read the relevant sections of the file again and extend <old_text> to be more specific."
            )
            raise ModelRetry(msg)

        logger.info("Applied %s/%s structured edits", applied_edits, len(old_texts))
        return content


def get_changed_lines(original_content: str, new_content: str, path: str) -> list[str]:
    diff = list(
        difflib.unified_diff(
            original_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
            lineterm="",
        )
    )
    return [line for line in diff if line.startswith(("+", "-"))]
