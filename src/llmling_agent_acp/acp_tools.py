"""ACP capability-based resource provider."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext  # noqa: TC002

from acp.acp_types import PlanEntryPriority, PlanEntryStatus  # noqa: TC001
from acp.schema import (
    AgentPlan,
    ContentToolCallContent,
    CreateTerminalRequest,
    EnvVariable,
    KillTerminalCommandRequest,
    PlanEntry,
    ReadTextFileRequest,
    ReleaseTerminalRequest,
    SessionNotification,
    TerminalOutputRequest,
    TerminalToolCallContent,
    TextContentBlock,
    ToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
    WaitForTerminalExitRequest,
    WriteTextFileRequest,
)
from llmling_agent.log import get_logger
from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from acp.schema import ClientCapabilities
    from llmling_agent_acp.acp_agent import LLMlingACPAgent


logger = get_logger(__name__)


class ACPCapabilityResourceProvider(ResourceProvider):
    """Provides ACP client-side tools based on capabilities and session context.

    This provider creates session-aware tools for terminal operations, filesystem
    access, and other ACP client capabilities. All tools have the session ID
    baked in at creation time, eliminating the need for parameter injection.
    """

    def __init__(
        self,
        agent: LLMlingACPAgent,
        session_id: str,
        client_capabilities: ClientCapabilities,
        cwd: str | None = None,
    ):
        """Initialize capability provider.

        Args:
            agent: The ACP agent instance
            session_id: Session ID for all tools created by this provider
            client_capabilities: Client-reported capabilities
            cwd: Current working directory for relative path resolution
        """
        super().__init__(name=f"acp_capabilities_{session_id}")
        self.agent = agent
        self.session_id = session_id
        self.client_capabilities = client_capabilities
        self.cwd = cwd
        self._current_plan: list[PlanEntry] = []

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
        """Get tools based on client capabilities."""
        tools: list[Tool] = []

        # Plan tools - always available for ACP sessions
        tools.extend(self._get_plan_tools())

        # Terminal tools if supported by both client and agent
        if self.client_capabilities.terminal and self.agent.terminal_access:
            tools.extend(self._get_terminal_tools())

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

    def _get_terminal_tools(self) -> list[Tool]:
        """Get all terminal tools with session_id baked in."""
        return [
            Tool.from_callable(
                self._create_run_command_tool(),
                source="terminal",
                name_override="run_command",
            ),
            Tool.from_callable(
                self._create_get_command_output_tool(),
                source="terminal",
                name_override="get_command_output",
            ),
            Tool.from_callable(
                self._create_create_terminal_tool(),
                source="terminal",
                name_override="create_terminal",
            ),
            Tool.from_callable(
                self._create_wait_for_terminal_exit_tool(),
                source="terminal",
                name_override="wait_for_terminal_exit",
            ),
            Tool.from_callable(
                self._create_kill_terminal_tool(),
                source="terminal",
                name_override="kill_terminal",
            ),
            Tool.from_callable(
                self._create_release_terminal_tool(),
                source="terminal",
                name_override="release_terminal",
            ),
        ]

    # Terminal Tool Implementations

    def _create_run_command_tool(self):
        """Create a tool that runs commands with live terminal output."""

        async def run_command(  # noqa: D417
            ctx: RunContext[Any],
            command: str,
            args: list[str] | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            output_char_limit: int | None = None,
            timeout_seconds: int | None = None,
        ) -> str:
            r"""Execute a shell command with live terminal output visible to the user.

            This tool shows real-time command output in the UI as it happens,
            perfect for long-running commands like builds, tests, or installations.

            Args:
                command: The command to execute (e.g., 'npm', 'python', 'make')
                args: Command arguments as list (e.g., ['test', '--coverage'])
                cwd: Working directory path
                env: Environment variables as key-value pairs
                output_char_limit: Maximum output characters to retain
                timeout_seconds: Maximum time to wait for command completion

            Returns:
                Command completion status and any final output
            """
            # Send initial tool call notification
            assert ctx.tool_call_id, (
                "Tool call ID must be present for terminal operations"
            )

            try:
                # Create terminal
                create_request = CreateTerminalRequest(
                    session_id=self.session_id,
                    command=command,
                    args=args or [],
                    cwd=cwd,
                    env=[EnvVariable(name=k, value=v) for k, v in (env or {}).items()],
                    output_byte_limit=output_char_limit,
                )

                create_response = await self.agent.connection.create_terminal(
                    create_request
                )
                terminal_id = create_response.terminal_id

                # Send initial notification with embedded terminal for live output
                cmd_display = f"{command} {' '.join(args or [])}"
                initial_update = SessionNotification(
                    session_id=self.session_id,
                    update=ToolCallStart(
                        tool_call_id=ctx.tool_call_id,
                        status="pending",
                        title=f"Running: {cmd_display}",
                        kind="execute",
                        content=[TerminalToolCallContent(terminal_id=terminal_id)],
                    ),
                )
                await self.agent.connection.session_update(initial_update)

                # Wait for completion (with optional timeout)
                wait_request = WaitForTerminalExitRequest(
                    session_id=self.session_id,
                    terminal_id=terminal_id,
                )

                if timeout_seconds:
                    import asyncio

                    try:
                        exit_result = await asyncio.wait_for(
                            self.agent.connection.wait_for_terminal_exit(wait_request),
                            timeout=timeout_seconds,
                        )
                    except TimeoutError:
                        # Kill the command on timeout
                        kill_request = KillTerminalCommandRequest(
                            session_id=self.session_id,
                            terminal_id=terminal_id,
                        )
                        await self.agent.connection.kill_terminal(kill_request)

                        # Get output before releasing
                        output_request = TerminalOutputRequest(
                            session_id=self.session_id,
                            terminal_id=terminal_id,
                        )
                        output_response = await self.agent.connection.terminal_output(
                            output_request
                        )

                        # Send timeout notification
                        timeout_update = SessionNotification(
                            session_id=self.session_id,
                            update=ToolCallProgress(
                                tool_call_id=ctx.tool_call_id,
                                status="failed",
                                title=f"Command timed out after {timeout_seconds}s",
                                content=[
                                    TerminalToolCallContent(terminal_id=terminal_id)
                                ],
                            ),
                        )
                        await self.agent.connection.session_update(timeout_update)

                        # Release terminal
                        release_request = ReleaseTerminalRequest(
                            session_id=self.session_id,
                            terminal_id=terminal_id,
                        )
                        await self.agent.connection.release_terminal(release_request)

                        return (
                            f"Command timed out after {timeout_seconds} seconds. "
                            f"Output: \n{output_response.output}"
                        )
                else:
                    exit_result = await self.agent.connection.wait_for_terminal_exit(
                        wait_request
                    )

                # Get final output
                output_request = TerminalOutputRequest(
                    session_id=self.session_id,
                    terminal_id=terminal_id,
                )
                output_response = await self.agent.connection.terminal_output(
                    output_request
                )

                # Send completion notification (terminal remains embedded for viewing)
                status = "completed" if (exit_result.exit_code or 0) == 0 else "failed"
                exit_code = exit_result.exit_code or 0

                completion_update = SessionNotification(
                    session_id=self.session_id,
                    update=ToolCallProgress(
                        tool_call_id=ctx.tool_call_id,
                        status=status,
                        title=f"Command completed (exit code: {exit_code})",
                        content=[TerminalToolCallContent(terminal_id=terminal_id)],
                    ),
                )
                await self.agent.connection.session_update(completion_update)

                # Release terminal (output remains visible in UI)
                release_request = ReleaseTerminalRequest(
                    session_id=self.session_id,
                    terminal_id=terminal_id,
                )
                await self.agent.connection.release_terminal(release_request)
                result = f"Command completed with exit code {exit_code}:\n"
                result += f"Output:\n{output_response.output}"
                if output_response.truncated:
                    result += " (output was truncated)"
            except Exception as e:  # noqa: BLE001
                # Send error notification
                error_update = SessionNotification(
                    session_id=self.session_id,
                    update=ToolCallProgress(
                        tool_call_id=ctx.tool_call_id,
                        status="failed",
                        title=f"Command failed: {e}",
                    ),
                )
                try:
                    await self.agent.connection.session_update(error_update)
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to send error update")

                return f"Error executing command: {e}"
            else:
                return result

        return run_command

    def _create_get_command_output_tool(self):
        """Create a tool that gets output from a running command."""

        async def get_command_output(ctx: RunContext[Any], terminal_id: str) -> str:  # noqa: D417
            """Get output from a terminal that was created with create_terminal.

            Use this to check output from a terminal created separately.
            Do NOT use this for simple command execution - use run_command instead.

            Args:
                terminal_id: The terminal ID returned by create_terminal

            Returns:
                Current output and status (running/completed/failed)
            """
            request = TerminalOutputRequest(
                session_id=self.session_id,
                terminal_id=terminal_id,
            )
            try:
                output_response = await self.agent.connection.terminal_output(request)
                result = output_response.output
                if output_response.truncated:
                    result += "\n[Output was truncated]"
                if output_response.exit_status:
                    if (code := output_response.exit_status.exit_code) is not None:
                        result += f"\n[Exited with code {code}]"
                    if signal := output_response.exit_status.signal:
                        result += f"\n[Terminated by signal {signal}]"
                else:
                    result += "\n[Still running]"
            except Exception as e:  # noqa: BLE001
                return f"Error getting command output: {e}"
            else:
                return result

        return get_command_output

    def _create_create_terminal_tool(self):
        """Create a tool that creates a terminal and returns the terminal ID."""

        async def create_terminal(  # noqa: D417
            ctx: RunContext[Any],
            command: str,
            args: list[str] | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            output_byte_limit: int = 1048576,
        ) -> str:
            """Create a terminal for advanced terminal management (ADVANCED USE ONLY).

            WARNING: This only creates a terminal and returns its ID. It does NOT wait for
            completion or get output. You must use get_command_output,
            wait_for_terminal_exit, and release_terminal separately.

            For simple command execution, use run_command instead - it's much easier.

            Args:
                command: The command to execute
                args: Command arguments
                cwd: Working directory
                env: Environment variables
                output_byte_limit: Maximum output bytes to retain

            Returns:
                Terminal ID (you must manage this terminal manually)
            """
            logger.info("🔧 DEBUG: Starting create_terminal execution")
            logger.info("🔧 DEBUG: Command: %s, Args: %s", command, args)

            request = CreateTerminalRequest(
                session_id=self.session_id,
                command=command,
                args=args or [],
                cwd=cwd,
                env=[EnvVariable(name=k, value=v) for k, v in (env or {}).items()],
                output_byte_limit=output_byte_limit,
            )
            logger.info("🔧 DEBUG: Calling create_terminal")
            try:
                create_response = await self.agent.connection.create_terminal(request)
                logger.info("🔧 DEBUG: Got create_response: %s", create_response)
                terminal_id = create_response.terminal_id
            except Exception as e:
                logger.exception("🔧 DEBUG: Exception in create_terminal")
                return f"Error creating terminal: {e}"
            else:
                logger.info("🔧 DEBUG: Returning terminal_id: %s", terminal_id)
                return terminal_id

        return create_terminal

    def _create_wait_for_terminal_exit_tool(self):
        """Create a tool that waits for a terminal to exit."""

        async def wait_for_terminal_exit(ctx: RunContext[Any], terminal_id: str) -> str:  # noqa: D417
            """Wait for a terminal to finish (ADVANCED USE ONLY).

            Only use this with terminals created by create_terminal.
            For simple command execution, use run_command instead.

            Args:
                terminal_id: The terminal ID from create_terminal

            Returns:
                Exit status information
            """
            request = WaitForTerminalExitRequest(
                session_id=self.session_id,
                terminal_id=terminal_id,
            )
            try:
                exit_response = await self.agent.connection.wait_for_terminal_exit(
                    request
                )

                result = f"Terminal {terminal_id} completed"
                if exit_response.exit_code is not None:
                    result += f" with exit code {exit_response.exit_code}"
                if exit_response.signal:
                    result += f" (terminated by signal {exit_response.signal})"
            except Exception as e:  # noqa: BLE001
                return f"Error waiting for terminal exit: {e}"
            else:
                return result

        return wait_for_terminal_exit

    def _create_kill_terminal_tool(self):
        """Create a tool that kills a running terminal command."""

        async def kill_terminal(ctx: RunContext[Any], terminal_id: str) -> str:  # noqa: D417
            """Forcefully stop a running terminal (ADVANCED USE ONLY).

            Only use this with terminals created by create_terminal.

            Args:
                terminal_id: The terminal ID from create_terminal

            Returns:
                Success/failure message
            """
            request = KillTerminalCommandRequest(
                session_id=self.session_id,
                terminal_id=terminal_id,
            )
            try:
                await self.agent.connection.kill_terminal(request)
            except Exception as e:  # noqa: BLE001
                return f"Error killing terminal: {e}"
            else:
                return f"Terminal {terminal_id} killed successfully"

        return kill_terminal

    def _create_release_terminal_tool(self):
        """Create a tool that releases terminal resources."""

        async def release_terminal(ctx: RunContext[Any], terminal_id: str) -> str:  # noqa: D417
            """Clean up a terminal created with create_terminal (ADVANCED USE ONLY).

            Only use this with terminals created by create_terminal.
            The run_command tool handles cleanup automatically.

            Args:
                terminal_id: The terminal ID from create_terminal

            Returns:
                Success/failure message
            """
            request = ReleaseTerminalRequest(
                session_id=self.session_id,
                terminal_id=terminal_id,
            )
            try:
                await self.agent.connection.release_terminal(request)
            except Exception as e:  # noqa: BLE001
                return f"Error releasing terminal: {e}"
            else:
                return f"Terminal {terminal_id} released successfully"

        return release_terminal

    def _create_run_command_with_timeout_tool(self):
        """Create a tool that runs commands with timeout support."""

        async def run_command_with_timeout(  # noqa: D417
            ctx: RunContext[Any],
            command: str,
            args: list[str] | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_seconds: int = 30,
            output_char_limit: int | None = None,
        ) -> str:
            """Execute a command with automatic timeout protection.

            Use this when running commands that might hang or take too long.
            Like run_command but will automatically kill the command if it exceeds
            the timeout.

            Args:
                command: The command to execute
                args: Command arguments
                cwd: Working directory
                env: Environment variables
                timeout_seconds: Maximum time to wait before killing (default: 30)
                output_char_limit: Maximum number of characters to capture from output

            Returns:
                Command output, or timeout message if command was killed
            """
            create_request = CreateTerminalRequest(
                session_id=self.session_id,
                command=command,
                args=args or [],
                cwd=cwd,
                env=[EnvVariable(name=k, value=v) for k, v in (env or {}).items()],
                output_byte_limit=output_char_limit,
            )
            try:
                create_response = await self.agent.connection.create_terminal(
                    create_request
                )
                terminal_id = create_response.terminal_id
                wait_request = WaitForTerminalExitRequest(
                    session_id=self.session_id,
                    terminal_id=terminal_id,
                )
                try:
                    await asyncio.wait_for(
                        self.agent.connection.wait_for_terminal_exit(wait_request),
                        timeout=timeout_seconds,
                    )

                    # Get output
                    out_request = TerminalOutputRequest(
                        session_id=self.session_id,
                        terminal_id=terminal_id,
                    )
                    output_response = await self.agent.connection.terminal_output(
                        out_request
                    )

                    result = output_response.output
                    if output_response.exit_status:
                        code = output_response.exit_status.exit_code
                        if code is not None:
                            result += f"\n[Command exited with code {code}]"
                        if output_response.exit_status.signal:
                            signal = output_response.exit_status.signal
                            result += f"\n[Terminated by signal {signal}]"

                except TimeoutError:
                    # Kill the command on timeout
                    kill_request = KillTerminalCommandRequest(
                        session_id=self.session_id,
                        terminal_id=terminal_id,
                    )
                    try:
                        await self.agent.connection.kill_terminal(kill_request)

                        # Get partial output
                        request = TerminalOutputRequest(
                            session_id=self.session_id,
                            terminal_id=terminal_id,
                        )
                        output_response = await self.agent.connection.terminal_output(
                            request
                        )

                        result = output_response.output
                        timeout_msg = (
                            f"Command timed out after {timeout_seconds} "
                            f"seconds and was killed"
                        )
                        result += f"\n[{timeout_msg}]"
                    except Exception:  # noqa: BLE001
                        result = (
                            f"Command timed out after {timeout_seconds} "
                            f"seconds and failed to kill"
                        )

                finally:
                    # Always release terminal
                    release_request = ReleaseTerminalRequest(
                        session_id=self.session_id,
                        terminal_id=terminal_id,
                    )
                    with contextlib.suppress(Exception):
                        await self.agent.connection.release_terminal(release_request)

            except Exception as e:  # noqa: BLE001
                return f"Error executing command with timeout: {e}"

            return result

        return run_command_with_timeout

    # Filesystem Tool Implementations

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

    def _get_plan_tools(self) -> list[Tool]:
        """Get plan management tools."""
        return [
            Tool.from_callable(
                self._create_add_plan_entry_tool(),
                source="planning",
                name_override="add_plan_entry",
            ),
            Tool.from_callable(
                self._create_update_plan_entry_tool(),
                source="planning",
                name_override="update_plan_entry",
            ),
            Tool.from_callable(
                self._create_remove_plan_entry_tool(),
                source="planning",
                name_override="remove_plan_entry",
            ),
        ]

    def _create_add_plan_entry_tool(self):
        """Create add plan entry tool."""

        async def add_plan_entry(
            content: str,
            priority: PlanEntryPriority = "medium",
            index: int | None = None,
        ) -> str:
            """Add a new plan entry.

            Args:
                content: Description of what this task aims to accomplish
                priority: Relative importance (high/medium/low)
                index: Optional position to insert at (default: append to end)

            Returns:
                Success message indicating entry was added
            """
            entry = PlanEntry(
                content=content,
                priority=priority,
                status="pending",
            )

            if index is None:
                self._current_plan.append(entry)
                entry_index = len(self._current_plan) - 1
            else:
                if index < 0 or index > len(self._current_plan):
                    return (
                        f"Error: Index {index} out of range (0-{len(self._current_plan)})"
                    )
                self._current_plan.insert(index, entry)
                entry_index = index

            # Send plan update
            await self._send_plan_update()

            return (
                f"Added plan entry at index {entry_index}: "
                f"'{content}' (priority: {priority})"
            )

        return add_plan_entry

    def _create_update_plan_entry_tool(self):
        """Create update plan entry tool."""

        async def update_plan_entry(
            index: int,
            content: str | None = None,
            status: PlanEntryStatus | None = None,
            priority: PlanEntryPriority | None = None,
        ) -> str:
            """Update an existing plan entry.

            Args:
                index: Position of entry to update (0-based)
                content: New task description
                status: New execution status
                priority: New priority level

            Returns:
                Success message indicating what was updated
            """
            if index < 0 or index >= len(self._current_plan):
                return (
                    f"Error: Index {index} out of range (0-{len(self._current_plan) - 1})"
                )

            entry = self._current_plan[index]
            updates = []

            if content is not None:
                entry.content = content
                updates.append(f"content to '{content}'")

            if status is not None:
                entry.status = status
                updates.append(f"status to '{status}'")

            if priority is not None:
                entry.priority = priority
                updates.append(f"priority to '{priority}'")

            if not updates:
                return "No changes specified"

            # Send plan update
            await self._send_plan_update()

            return f"Updated entry {index}: {', '.join(updates)}"

        return update_plan_entry

    def _create_remove_plan_entry_tool(self):
        """Create remove plan entry tool."""

        async def remove_plan_entry(index: int) -> str:
            """Remove a plan entry.

            Args:
                index: Position of entry to remove (0-based)

            Returns:
                Success message indicating entry was removed
            """
            if index < 0 or index >= len(self._current_plan):
                return (
                    f"Error: Index {index} out of range (0-{len(self._current_plan) - 1})"
                )

            removed_entry = self._current_plan.pop(index)

            # Send plan update
            await self._send_plan_update()

            if self._current_plan:
                return (
                    f"Removed entry {index}: '{removed_entry.content}', "
                    f"remaining entries reindexed"
                )
            return f"Removed entry {index}: '{removed_entry.content}', plan is now empty"

        return remove_plan_entry

    async def _send_plan_update(self):
        """Send current plan state via session update."""
        if not self._current_plan:
            # Don't send empty plans
            return

        plan = AgentPlan(entries=self._current_plan)
        notification = SessionNotification(session_id=self.session_id, update=plan)
        await self.agent.connection.session_update(notification)
