"""ACP terminal provider for shell command execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext  # noqa: TC002

from acp.schema import (
    ContentToolCallContent,
    CreateTerminalRequest,
    EnvVariable,
    KillTerminalCommandRequest,
    ReleaseTerminalRequest,
    SessionNotification,
    TerminalOutputRequest,
    TerminalToolCallContent,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    WaitForTerminalExitRequest,
)
from llmling_agent.log import get_logger
from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from acp.schema import ClientCapabilities
    from llmling_agent_acp.session import ACPSession


logger = get_logger(__name__)


class ACPTerminalProvider(ResourceProvider):
    """Provides ACP terminal-related tools for command execution.

    This provider creates session-aware tools for executing shell commands
    via the ACP client. All tools have the session ID baked in at creation time,
    eliminating the need for parameter injection.
    """

    def __init__(
        self,
        session: ACPSession,
        client_capabilities: ClientCapabilities,
        cwd: str | None = None,
    ):
        """Initialize terminal provider.

        Args:
            session: Session for all tools created by this provider
            client_capabilities: Client-reported capabilities
            cwd: Current working directory for relative path resolution
        """
        super().__init__(name=f"acp_terminal_{session.session_id}")
        self.agent = session.acp_agent
        self.session_id = session.session_id
        self.session = session
        self.client_capabilities = client_capabilities
        self.cwd = cwd

    async def get_tools(self) -> list[Tool]:
        """Get all terminal tools with session_id baked in."""
        if self.client_capabilities.terminal and self.agent.terminal_access:
            return [
                Tool.from_callable(
                    self.run_command, source="terminal", category="execute"
                ),
                Tool.from_callable(
                    self.get_command_output, source="terminal", category="read"
                ),
                Tool.from_callable(
                    self.create_terminal, source="terminal", category="execute"
                ),
                Tool.from_callable(
                    self.wait_for_terminal_exit, source="terminal", category="execute"
                ),
                Tool.from_callable(
                    self.kill_terminal, source="terminal", category="execute"
                ),
                Tool.from_callable(
                    self.release_terminal, source="terminal", category="execute"
                ),
            ]

        return []

    async def run_command(  # noqa: D417
        self,
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
        assert ctx.tool_call_id, "Tool call ID must be present for terminal operations"
        create_request = CreateTerminalRequest(
            session_id=self.session_id,
            command=command,
            args=args or [],
            cwd=cwd,
            env=[EnvVariable(name=k, value=v) for k, v in (env or {}).items()],
            output_byte_limit=output_char_limit,
        )
        try:
            create_response = await self.agent.connection.create_terminal(create_request)
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
                            content=[TerminalToolCallContent(terminal_id=terminal_id)],
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
            output_response = await self.agent.connection.terminal_output(output_request)

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

    async def get_command_output(self, ctx: RunContext[Any], terminal_id: str) -> str:  # noqa: D417
        """Get output from a terminal that was created with create_terminal.

        Use this to check output from a terminal created separately.
        Do NOT use this for simple command execution - use run_command instead.

        Args:
            terminal_id: The terminal ID returned by create_terminal

        Returns:
            Current output and status (running/completed/failed)
        """
        # Send initial pending notification
        assert ctx.tool_call_id, "Tool call ID must be present for terminal operations"
        start = ToolCallStart(
            tool_call_id=ctx.tool_call_id,
            status="pending",
            title=f"Getting output from terminal {terminal_id}",
            kind="read",
        )
        try:
            notifi = SessionNotification(session_id=self.session_id, update=start)
            await self.agent.connection.session_update(notifi)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to send pending update: %s", e)

        req = TerminalOutputRequest(session_id=self.session_id, terminal_id=terminal_id)
        try:
            output_response = await self.agent.connection.terminal_output(req)

            # Determine status for both UI and agent
            exit_code = None
            signal = None
            status_text = "Running"

            if output_response.exit_status:
                exit_code = output_response.exit_status.exit_code
                signal = output_response.exit_status.signal
                if exit_code is not None:
                    status_text = f"Exited with code {exit_code}"
                elif signal:
                    status_text = f"Terminated by signal {signal}"

            # Send completion update with output as content
            content_block = TextContentBlock(text=output_response.output)
            progress = ToolCallProgress(
                tool_call_id=ctx.tool_call_id,
                status="completed",
                title=f"Terminal {terminal_id} output retrieved",
                content=[ContentToolCallContent(content=content_block)],
                raw_output=f"Retrieved output from terminal {terminal_id}",
            )
            notifi = SessionNotification(session_id=self.session_id, update=progress)
            await self.agent.connection.session_update(notifi)

            # Return structured result for agent
            result = f"""Terminal {terminal_id} Output:
{output_response.output}

Status: {status_text}"""

            if output_response.truncated:
                result += "\n(output truncated)"
        except Exception as e:  # noqa: BLE001
            # Send failed update
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
            return f"Error getting command output: {e}"
        else:
            return result

    async def create_terminal(  # noqa: D417
        self,
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
            Terminal creation details with ID for management
        """
        # Send initial pending notification
        assert ctx.tool_call_id, "Tool call ID must be present for terminal operations"
        cmd_display = f"{command} {' '.join(args or [])}"
        start = ToolCallStart(
            tool_call_id=ctx.tool_call_id,
            status="pending",
            title=f"Creating terminal: {cmd_display}",
            kind="execute",
        )
        try:
            notifi = SessionNotification(session_id=self.session_id, update=start)
            await self.agent.connection.session_update(notifi)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to send pending update: %s", e)

        request = CreateTerminalRequest(
            session_id=self.session_id,
            command=command,
            args=args or [],
            cwd=cwd,
            env=[EnvVariable(name=k, value=v) for k, v in (env or {}).items()],
            output_byte_limit=output_byte_limit,
        )

        try:
            create_response = await self.agent.connection.create_terminal(request)
            terminal_id = create_response.terminal_id

            # Send completion update
            progress = ToolCallProgress(
                tool_call_id=ctx.tool_call_id,
                status="completed",
                title=f"Terminal {terminal_id} created",
                raw_output=f"Created terminal: {terminal_id}",
            )
            notifi = SessionNotification(session_id=self.session_id, update=progress)
            await self.agent.connection.session_update(notifi)
        except Exception as e:  # noqa: BLE001
            # Send failed update
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
            return f"Error creating terminal: {e}"
        else:
            return f"""Created terminal: {terminal_id}
Command: {cmd_display}
Working Directory: {cwd or "default"}
Output Limit: {output_byte_limit} bytes
Status: Running

Use this terminal_id with other terminal tools:
- wait_for_terminal_exit({terminal_id})
- get_command_output({terminal_id})
- kill_terminal({terminal_id})
- release_terminal({terminal_id})"""

    async def wait_for_terminal_exit(self, ctx: RunContext[Any], terminal_id: str) -> str:  # noqa: D417
        """Wait for a terminal to finish (ADVANCED USE ONLY).

        Only use this with terminals created by create_terminal.
        For simple command execution, use run_command instead.

        Args:
            terminal_id: The terminal ID from create_terminal

        Returns:
            Exit status information and final output
        """
        # Send initial notification with embedded terminal for live view
        assert ctx.tool_call_id, "Tool call ID must be present for terminal operations"
        start = ToolCallStart(
            tool_call_id=ctx.tool_call_id,
            status="pending",
            title=f"Waiting for terminal {terminal_id} to complete",
            kind="execute",
            content=[TerminalToolCallContent(terminal_id=terminal_id)],
        )
        try:
            notifi = SessionNotification(session_id=self.session_id, update=start)
            await self.agent.connection.session_update(notifi)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to send pending update: %s", e)

        request = WaitForTerminalExitRequest(
            session_id=self.session_id,
            terminal_id=terminal_id,
        )
        try:
            exit_response = await self.agent.connection.wait_for_terminal_exit(request)

            output_request = TerminalOutputRequest(
                session_id=self.session_id,
                terminal_id=terminal_id,
            )
            output_response = await self.agent.connection.terminal_output(output_request)

            # Send completion notification (terminal remains embedded for viewing)
            exit_code = exit_response.exit_code or 0
            status = "completed" if exit_code == 0 else "failed"

            completion_update = SessionNotification(
                session_id=self.session_id,
                update=ToolCallProgress(
                    tool_call_id=ctx.tool_call_id,
                    status=status,
                    title=f"Terminal completed (exit code: {exit_code})",
                    content=[TerminalToolCallContent(terminal_id=terminal_id)],
                ),
            )
            await self.agent.connection.session_update(completion_update)

            # Return structured result for agent
            result = f"""Terminal {terminal_id} completed
Exit Code: {exit_code}
Status: {"SUCCESS" if exit_code == 0 else "FAILED"}"""

            if exit_response.signal:
                result += f"\nTerminated by signal: {exit_response.signal}"

            result += f"\n\nFinal Output:\n{output_response.output}"

            if output_response.truncated:
                result += "\n(output truncated)"
        except Exception as e:  # noqa: BLE001
            # Send failed update
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
            return f"Error waiting for terminal exit: {e}"
        else:
            return result

    async def kill_terminal(self, ctx: RunContext[Any], terminal_id: str) -> str:  # noqa: D417
        """Kill a running terminal command (ADVANCED USE ONLY).

        Only use this with terminals created by create_terminal.

        Args:
            terminal_id: The terminal ID from create_terminal

        Returns:
            Kill confirmation
        """
        # Send initial pending notification
        assert ctx.tool_call_id, "Tool call ID must be present for terminal operations"
        try:
            start = ToolCallStart(
                tool_call_id=ctx.tool_call_id,
                status="pending",
                title=f"Killing terminal {terminal_id}",
                kind="execute",
            )
            notifi = SessionNotification(session_id=self.session_id, update=start)
            await self.agent.connection.session_update(notifi)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to send pending update: %s", e)

        request = KillTerminalCommandRequest(
            session_id=self.session_id,
            terminal_id=terminal_id,
        )
        try:
            await self.agent.connection.kill_terminal(request)

            # Send completion update
            progress = ToolCallProgress(
                tool_call_id=ctx.tool_call_id,
                status="completed",
                title=f"Terminal {terminal_id} killed",
                raw_output=f"Terminal {terminal_id} killed successfully",
            )
            notifi = SessionNotification(session_id=self.session_id, update=progress)
            await self.agent.connection.session_update(notifi)
        except Exception as e:  # noqa: BLE001
            # Send failed update
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
            return f"Error killing terminal: {e}"
        else:
            return f"Terminal {terminal_id} killed successfully"

    async def release_terminal(self, ctx: RunContext[Any], terminal_id: str) -> str:  # noqa: D417
        """Release a terminal session (ADVANCED USE ONLY).

        This cleans up a terminal created by create_terminal.
        Usually you don't need to call this directly as other tools
        handle cleanup automatically.

        Args:
            terminal_id: The terminal ID from create_terminal

        Returns:
            Release confirmation
        """
        # Send initial pending notification
        assert ctx.tool_call_id, "Tool call ID must be present for terminal operations"
        try:
            start = ToolCallStart(
                tool_call_id=ctx.tool_call_id,
                status="pending",
                title=f"Releasing terminal {terminal_id}",
                kind="execute",
            )
            notifi = SessionNotification(session_id=self.session_id, update=start)
            await self.agent.connection.session_update(notifi)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to send pending update: %s", e)

        request = ReleaseTerminalRequest(
            session_id=self.session_id,
            terminal_id=terminal_id,
        )
        try:
            await self.agent.connection.release_terminal(request)

            # Send completion update
            progress = ToolCallProgress(
                tool_call_id=ctx.tool_call_id,
                status="completed",
                title=f"Terminal {terminal_id} released",
                raw_output=f"Terminal {terminal_id} released successfully",
            )
            notifi = SessionNotification(session_id=self.session_id, update=progress)
            await self.agent.connection.session_update(notifi)
        except Exception as e:  # noqa: BLE001
            # Send failed update
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
            return f"Error releasing terminal: {e}"
        else:
            return f"Terminal {terminal_id} released successfully"
