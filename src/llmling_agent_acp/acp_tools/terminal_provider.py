"""ACP terminal provider for shell command execution."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext  # noqa: TC002

from acp.schema import (
    CreateTerminalRequest,
    EnvVariable,
    KillTerminalCommandRequest,
    ReleaseTerminalRequest,
    SessionNotification,
    TerminalOutputRequest,
    TerminalToolCallContent,
    ToolCallProgress,
    ToolCallStart,
    WaitForTerminalExitRequest,
)
from llmling_agent.log import get_logger
from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from acp.schema import ClientCapabilities
    from llmling_agent_acp.acp_agent import LLMlingACPAgent


logger = get_logger(__name__)


class ACPTerminalProvider(ResourceProvider):
    """Provides ACP terminal-related tools for command execution.

    This provider creates session-aware tools for executing shell commands
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
        """Initialize terminal provider.

        Args:
            agent: The ACP agent instance
            session_id: Session ID for all tools created by this provider
            client_capabilities: Client-reported capabilities
            cwd: Current working directory for relative path resolution
        """
        super().__init__(name=f"acp_terminal_{session_id}")
        self.agent = agent
        self.session_id = session_id
        self.client_capabilities = client_capabilities
        self.cwd = cwd

    async def get_tools(self) -> list[Tool]:
        """Get all terminal tools with session_id baked in."""
        if self.client_capabilities.terminal and self.agent.terminal_access:
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

        return []

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
