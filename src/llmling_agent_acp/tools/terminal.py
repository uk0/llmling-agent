"""Terminal tools for ACP agent client-side command execution."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from acp.schema import (
    CreateTerminalRequest,
    EnvVariable,
    KillTerminalCommandRequest,
    ReleaseTerminalRequest,
    TerminalOutputRequest,
    WaitForTerminalExitRequest,
)
from llmling_agent.log import get_logger
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from llmling_agent_acp.acp_agent import LLMlingACPAgent


logger = get_logger(__name__)


def get_terminal_tools(agent: LLMlingACPAgent) -> list[Tool]:
    """Get terminal tools for ACP agent.

    Args:
        agent: The ACP agent instance

    Returns:
        List of terminal tools
    """
    return [
        Tool.from_callable(
            _create_run_command_tool(agent),
            source="terminal",
            name_override="run_command",
        ),
        Tool.from_callable(
            _create_get_command_output_tool(agent),
            source="terminal",
            name_override="get_command_output",
        ),
        Tool.from_callable(
            _create_create_terminal_tool(agent),
            source="terminal",
            name_override="create_terminal",
        ),
        Tool.from_callable(
            _create_wait_for_terminal_exit_tool(agent),
            source="terminal",
            name_override="wait_for_terminal_exit",
        ),
        Tool.from_callable(
            _create_kill_terminal_tool(agent),
            source="terminal",
            name_override="kill_terminal",
        ),
        Tool.from_callable(
            _create_release_terminal_tool(agent),
            source="terminal",
            name_override="release_terminal",
        ),
        Tool.from_callable(
            _create_run_command_with_timeout_tool(agent),
            source="terminal",
            name_override="run_command_with_timeout",
        ),
    ]


def _create_run_command_tool(agent: LLMlingACPAgent):
    """Create a tool that runs commands via the ACP client."""

    async def run_command(
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        session_id: str = "default_session",
    ) -> str:
        """Execute a command in the client's environment.

        Args:
            command: The command to execute
            args: Command arguments (optional)
            cwd: Working directory (optional)
            env: Environment variables (optional)
            session_id: Session ID for the request

        Returns:
            Combined stdout and stderr output, or error message
        """
        try:
            # Create terminal via client
            env_var = [EnvVariable(name=k, value=v) for k, v in (env or {}).items()]
            create_request = CreateTerminalRequest(
                session_id=session_id,
                command=command,
                args=args or [],
                cwd=cwd,
                env=env_var,
                output_byte_limit=1048576,
            )
            create_response = await agent.connection.create_terminal(create_request)
            terminal_id = create_response.terminal_id

            # Wait for completion
            wait_request = WaitForTerminalExitRequest(
                session_id=session_id,
                terminal_id=terminal_id,
            )
            await agent.connection.wait_for_terminal_exit(wait_request)

            # Get output
            output_request = TerminalOutputRequest(
                session_id=session_id,
                terminal_id=terminal_id,
            )
            output_response = await agent.connection.terminal_output(output_request)

            # Release terminal
            release_request = ReleaseTerminalRequest(
                session_id=session_id,
                terminal_id=terminal_id,
            )
            await agent.connection.release_terminal(release_request)

            result = output_response.output
            if output_response.exit_status:
                code = output_response.exit_status.exit_code
                if code is not None:
                    result += f"\n[Command exited with code {code}]"
                if output_response.exit_status.signal:
                    result += (
                        f"\n[Terminated by signal {output_response.exit_status.signal}]"
                    )

        except Exception as e:  # noqa: BLE001
            return f"Error executing command: {e}"
        else:
            return result

    return run_command


def _create_get_command_output_tool(agent: LLMlingACPAgent):
    """Create a tool that gets output from a running command."""

    async def get_command_output(
        terminal_id: str,
        session_id: str = "default_session",
    ) -> str:
        """Get current output from a running command.

        Args:
            terminal_id: The terminal ID to get output from
            session_id: Session ID for the request

        Returns:
            Current command output
        """
        try:
            # Get output
            request = TerminalOutputRequest(
                session_id=session_id,
                terminal_id=terminal_id,
            )
            output_response = await agent.connection.terminal_output(request)

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


def _create_create_terminal_tool(agent: LLMlingACPAgent):
    """Create a tool that creates a terminal and returns the terminal ID."""

    async def create_terminal(
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        output_byte_limit: int = 1048576,
        session_id: str = "default_session",
    ) -> str:
        """Create a terminal and start executing a command.

        Args:
            command: The command to execute
            args: Command arguments (optional)
            cwd: Working directory (optional)
            env: Environment variables (optional)
            output_byte_limit: Maximum output bytes to retain
            session_id: Session ID for the request

        Returns:
            Terminal ID for the created terminal
        """
        try:
            request = CreateTerminalRequest(
                session_id=session_id,
                command=command,
                args=args or [],
                cwd=cwd,
                env=[EnvVariable(name=k, value=v) for k, v in (env or {}).items()],
                output_byte_limit=output_byte_limit,
            )
            create_response = await agent.connection.create_terminal(request)
        except Exception as e:  # noqa: BLE001
            return f"Error creating terminal: {e}"
        else:
            return create_response.terminal_id

    return create_terminal


def _create_wait_for_terminal_exit_tool(agent: LLMlingACPAgent):
    """Create a tool that waits for a terminal to exit."""

    async def wait_for_terminal_exit(
        terminal_id: str,
        session_id: str = "default_session",
    ) -> str:
        """Wait for a terminal command to complete.

        Args:
            terminal_id: The terminal ID to wait for
            session_id: Session ID for the request

        Returns:
            Exit status information
        """
        try:
            request = WaitForTerminalExitRequest(
                session_id=session_id,
                terminal_id=terminal_id,
            )
            exit_response = await agent.connection.wait_for_terminal_exit(request)

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


def _create_kill_terminal_tool(agent: LLMlingACPAgent):
    """Create a tool that kills a running terminal command."""

    async def kill_terminal(
        terminal_id: str,
        session_id: str = "default_session",
    ) -> str:
        """Kill a running terminal command.

        Args:
            terminal_id: The terminal ID to kill
            session_id: Session ID for the request

        Returns:
            Success/failure message
        """
        try:
            request = KillTerminalCommandRequest(
                session_id=session_id,
                terminal_id=terminal_id,
            )
            await agent.connection.kill_terminal(request)
        except Exception as e:  # noqa: BLE001
            return f"Error killing terminal: {e}"
        else:
            return f"Terminal {terminal_id} killed successfully"

    return kill_terminal


def _create_release_terminal_tool(agent: LLMlingACPAgent):
    """Create a tool that releases terminal resources."""

    async def release_terminal(
        terminal_id: str,
        session_id: str = "default_session",
    ) -> str:
        """Release a terminal and free its resources.

        Args:
            terminal_id: The terminal ID to release
            session_id: Session ID for the request

        Returns:
            Success/failure message
        """
        try:
            request = ReleaseTerminalRequest(
                session_id=session_id,
                terminal_id=terminal_id,
            )
            await agent.connection.release_terminal(request)
        except Exception as e:  # noqa: BLE001
            return f"Error releasing terminal: {e}"
        else:
            return f"Terminal {terminal_id} released successfully"

    return release_terminal


def _create_run_command_with_timeout_tool(agent: LLMlingACPAgent):
    """Create a tool that runs commands with timeout support."""

    async def run_command_with_timeout(
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: int = 30,
        session_id: str = "default_session",
    ) -> str:
        """Execute a command with timeout support.

        Args:
            command: The command to execute
            args: Command arguments (optional)
            cwd: Working directory (optional)
            env: Environment variables (optional)
            timeout_seconds: Timeout in seconds (default: 30)
            session_id: Session ID for the request

        Returns:
            Command output or timeout/error message
        """
        try:
            # Create terminal
            create_request = CreateTerminalRequest(
                session_id=session_id,
                command=command,
                args=args or [],
                cwd=cwd,
                env=[EnvVariable(name=k, value=v) for k, v in (env or {}).items()],
                output_byte_limit=1048576,
            )
            create_response = await agent.connection.create_terminal(create_request)
            terminal_id = create_response.terminal_id

            try:
                # Wait for completion with timeout
                wait_request = WaitForTerminalExitRequest(
                    session_id=session_id,
                    terminal_id=terminal_id,
                )
                await asyncio.wait_for(
                    agent.connection.wait_for_terminal_exit(wait_request),
                    timeout=timeout_seconds,
                )

                # Get output
                out_request = TerminalOutputRequest(
                    session_id=session_id,
                    terminal_id=terminal_id,
                )
                output_response = await agent.connection.terminal_output(out_request)

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
                try:
                    kill_request = KillTerminalCommandRequest(
                        session_id=session_id,
                        terminal_id=terminal_id,
                    )
                    await agent.connection.kill_terminal(kill_request)

                    # Get partial output
                    request = TerminalOutputRequest(
                        session_id=session_id,
                        terminal_id=terminal_id,
                    )
                    output_response = await agent.connection.terminal_output(request)

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
                    session_id=session_id,
                    terminal_id=terminal_id,
                )
                with contextlib.suppress(Exception):
                    await agent.connection.release_terminal(release_request)

        except Exception as e:  # noqa: BLE001
            return f"Error executing command with timeout: {e}"

        return result

    return run_command_with_timeout
