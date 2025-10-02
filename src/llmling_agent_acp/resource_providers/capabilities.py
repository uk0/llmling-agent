"""ACP capability-based resource provider."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext  # noqa: TC002

from acp.schema import (
    ContentToolCallContent,
    CreateTerminalRequest,
    EnvVariable,
    KillTerminalCommandRequest,
    ReadTextFileRequest,
    ReleaseTerminalRequest,
    SessionNotification,
    TerminalOutputRequest,
    TextContentBlock,
    ToolCallLocation,
    ToolCallProgress,
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
            Tool.from_callable(
                self._create_run_command_with_timeout_tool(),
                source="terminal",
                name_override="run_command_with_timeout",
            ),
        ]

    # Terminal Tool Implementations

    def _create_run_command_tool(self):
        """Create a tool that runs commands via the ACP client."""

        async def run_command(  # noqa: D417
            ctx: RunContext[Any],
            command: str,
            args: list[str] | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            output_char_limit: int | None = None,
        ) -> str:
            r"""Execute a shell command and return its output.

            Use this tool when you need to run a command and see its complete output.
            This handles the full lifecycle: creates terminal, runs command,
            waits for completion, captures output, and cleans up automatically.

            Args:
                command: The command to execute (e.g., 'echo', 'ls', 'python')
                args: Command arguments as list (e.g., ['hello world'] for echo)
                cwd: Working directory path
                env: Environment variables as key-value pairs
                output_char_limit: Maximum number of characters to capture from output

            Returns:
                Complete command output including stdout/stderr and exit status
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
                logger.info("🔧 DEBUG: Got create_response: %s", create_response)
                terminal_id = create_response.terminal_id
                logger.info("🔧 DEBUG: Extracted terminal_id: %s", terminal_id)

                # Wait for completion
                logger.info("🔧 DEBUG: Creating wait_request")
                wait_request = WaitForTerminalExitRequest(
                    session_id=self.session_id,
                    terminal_id=terminal_id,
                )
                logger.info("🔧 DEBUG: Calling wait_for_terminal_exit")
                await self.agent.connection.wait_for_terminal_exit(wait_request)
                logger.info("🔧 DEBUG: Terminal exit completed")

                # Get output
                logger.info("🔧 DEBUG: Getting terminal output")
                output_request = TerminalOutputRequest(
                    session_id=self.session_id,
                    terminal_id=terminal_id,
                )
                output_response = await self.agent.connection.terminal_output(
                    output_request
                )
                logger.info("🔧 DEBUG: Got output response")

                # Release terminal
                logger.info("🔧 DEBUG: Releasing terminal")
                release_request = ReleaseTerminalRequest(
                    session_id=self.session_id,
                    terminal_id=terminal_id,
                )
                await self.agent.connection.release_terminal(release_request)
                logger.info("🔧 DEBUG: Terminal released")

                result = output_response.output
                if output_response.exit_status:
                    code = output_response.exit_status.exit_code
                    if code is not None:
                        result += f"\n[Command exited with code {code}]"
                    if output_response.exit_status.signal:
                        signal = output_response.exit_status.signal
                        result += f"\n[Terminated by signal {signal}]"

            except Exception as e:  # noqa: BLE001
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
                try:
                    completed_update = SessionNotification(
                        session_id=self.session_id,
                        update=ToolCallProgress(
                            tool_call_id=ctx.tool_call_id,
                            status="completed",
                            locations=[ToolCallLocation(path=resolved_path)],
                            content=[
                                ContentToolCallContent(
                                    content=TextContentBlock(
                                        text=f"````\n{response.content}\n````"
                                    )
                                )
                            ],
                        ),
                    )
                    await self.agent.connection.session_update(completed_update)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to send completed update: %s", e)

            except Exception as e:  # noqa: BLE001
                # Send failed update
                assert ctx.tool_call_id, "Tool call ID must be present for fs operations"
                try:
                    failed_update = SessionNotification(
                        session_id=self.session_id,
                        update=ToolCallProgress(
                            tool_call_id=ctx.tool_call_id,
                            status="failed",
                            raw_output=f"Error: {e}",
                        ),
                    )
                    await self.agent.connection.session_update(failed_update)
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

            Use this to create configuration files, save data, write scripts,
            or update any text-based files on the client's filesystem.

            Args:
                path: File path (absolute or relative to session cwd)
                content: The complete text content to write to the file

            Returns:
                Success confirmation message, or error message if write fails
            """
            # Resolve relative paths against session cwd
            resolved_path = self._resolve_path(path)

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
                    completed_update = SessionNotification(
                        session_id=self.session_id,
                        update=ToolCallProgress(
                            tool_call_id=ctx.tool_call_id,
                            status="completed",
                            locations=[ToolCallLocation(path=resolved_path)],
                            content=[
                                ContentToolCallContent(
                                    content=TextContentBlock(
                                        text=f"Successfully wrote file: {path}"
                                    )
                                )
                            ],
                        ),
                    )
                    await self.agent.connection.session_update(completed_update)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to send completed update: %s", e)

            except Exception as e:  # noqa: BLE001
                # Send failed update
                assert ctx.tool_call_id, "Tool call ID must be present for fs operations"
                try:
                    failed_update = SessionNotification(
                        session_id=self.session_id,
                        update=ToolCallProgress(
                            tool_call_id=ctx.tool_call_id,
                            status="failed",
                            raw_output=f"Error: {e}",
                        ),
                    )
                    await self.agent.connection.session_update(failed_update)
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to send failed update")

                return f"Error writing file: {e}"
            else:
                return f"Successfully wrote file: {path}"

        return write_text_file
