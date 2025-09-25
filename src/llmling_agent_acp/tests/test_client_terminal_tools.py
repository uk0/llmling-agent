"""Tests for client-side terminal tools that make ACP requests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import pytest

from acp.schema import (
    CreateTerminalResponse,
    ReleaseTerminalResponse,
    TerminalExitStatus,
    TerminalOutputResponse,
    WaitForTerminalExitResponse,
)


if TYPE_CHECKING:
    from llmling_agent_acp.acp_agent import LLMlingACPAgent


class TestClientTerminalTools:
    """Test client-side terminal tools that request execution from ACP client."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock ACP connection."""
        from acp import AgentSideConnection

        return Mock(spec=AgentSideConnection)

    @pytest.fixture
    def mock_agent_pool(self):
        """Create a mock agent pool."""
        from llmling_agent import AgentPool

        pool = Mock(spec=AgentPool)
        pool.agents = {}
        return pool

    @pytest.fixture
    def acp_agent(self, mock_connection, mock_agent_pool):
        """Create ACP agent with terminal support."""
        from llmling_agent_acp.acp_agent import LLMlingACPAgent

        # Create mock agent
        mock_agent = Mock()
        mock_tools = {}

        def register_tool(tool):
            mock_tools[tool.name] = tool

        mock_agent.tools = Mock()
        mock_agent.tools.register_tool = register_tool
        mock_agent_pool.agents = {"test_agent": mock_agent}

        # Create ACP agent with terminal access
        agent = LLMlingACPAgent(
            connection=mock_connection,
            agent_pool=mock_agent_pool,
            terminal_access=True,
        )

        # Store reference to mock tools for testing
        agent._mock_tools = mock_tools  # type: ignore[attr-defined]
        return agent

    def test_terminal_tools_registered(self, acp_agent: LLMlingACPAgent):
        """Test that terminal tools are registered when terminal access is enabled."""
        expected_tools = {
            "run_command",
            "get_command_output",
            "create_terminal",
            "wait_for_terminal_exit",
            "kill_terminal",
            "release_terminal",
            "run_command_with_timeout",
        }
        registered_tools = set(acp_agent._mock_tools.keys())  # type: ignore[attr-defined]
        assert registered_tools == expected_tools

        for tool_name in expected_tools:
            tool = acp_agent._mock_tools[tool_name]  # type: ignore[attr-defined]
            assert tool.source == "terminal"

    def test_no_tools_without_terminal_access(self, mock_connection, mock_agent_pool):
        """Test no tools registered when terminal access disabled."""
        from llmling_agent_acp.acp_agent import LLMlingACPAgent

        mock_agent = Mock()
        mock_agent.tools = Mock()
        mock_agent.tools.register_tool = Mock()
        mock_agent_pool.agents = {"test_agent": mock_agent}

        LLMlingACPAgent(
            connection=mock_connection,
            agent_pool=mock_agent_pool,
            terminal_access=False,
        )

        mock_agent.tools.register_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_command_success(self, acp_agent: LLMlingACPAgent):
        """Test successful command execution."""
        # Mock ACP terminal operations on the connection
        acp_agent.connection.create_terminal = AsyncMock(  # type: ignore[method-assign]
            return_value=CreateTerminalResponse(terminal_id="term_123")
        )
        acp_agent.connection.wait_for_terminal_exit = AsyncMock(  # type: ignore[method-assign]
            return_value=WaitForTerminalExitResponse(exit_code=0, signal=None)
        )
        acp_agent.connection.terminal_output = AsyncMock(  # type: ignore[method-assign]
            return_value=TerminalOutputResponse(
                output="Hello, World!\n",
                truncated=False,
                exit_status=TerminalExitStatus(exit_code=0, signal=None),
            )
        )
        acp_agent.connection.release_terminal = AsyncMock(  # type: ignore[method-assign]
            return_value=ReleaseTerminalResponse()
        )

        # Execute run_command tool
        run_tool = acp_agent._mock_tools["run_command"]  # type: ignore[attr-defined]
        result = await run_tool.execute(
            command="echo",
            args=["Hello, World!"],
            session_id="test_session",
        )

        # Verify result
        assert "Hello, World!" in result
        assert "[Command exited with code 0]" in result

        # Verify ACP calls were made
        acp_agent.connection.create_terminal.assert_called_once()
        acp_agent.connection.wait_for_terminal_exit.assert_called_once()
        acp_agent.connection.terminal_output.assert_called_once()
        acp_agent.connection.release_terminal.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_command_failure(self, acp_agent: LLMlingACPAgent):
        """Test command execution with failure."""
        # Mock failed command
        acp_agent.connection.create_terminal = AsyncMock(  # type: ignore[method-assign]
            return_value=CreateTerminalResponse(terminal_id="term_456")
        )
        acp_agent.connection.wait_for_terminal_exit = AsyncMock(  # type: ignore[method-assign]
            return_value=WaitForTerminalExitResponse(exit_code=1, signal=None)
        )
        acp_agent.connection.terminal_output = AsyncMock(  # type: ignore[method-assign]
            return_value=TerminalOutputResponse(
                output="command not found\n",
                truncated=False,
                exit_status=TerminalExitStatus(exit_code=1, signal=None),
            )
        )
        acp_agent.connection.release_terminal = AsyncMock(  # type: ignore[method-assign]
            return_value=ReleaseTerminalResponse()
        )

        # Execute run_command tool
        run_tool = acp_agent._mock_tools["run_command"]  # type: ignore[attr-defined]
        result = await run_tool.execute(
            command="nonexistent_command",
            session_id="test_session",
        )

        # Verify error handling
        assert "command not found" in result
        assert "[Command exited with code 1]" in result

    @pytest.mark.asyncio
    async def test_get_command_output_running(self, acp_agent: LLMlingACPAgent):
        """Test getting output from running command."""
        # Mock running command
        acp_agent.connection.terminal_output = AsyncMock(  # type: ignore[method-assign]
            return_value=TerminalOutputResponse(
                output="Processing...\n",
                truncated=False,
                exit_status=None,  # Still running
            )
        )

        # Execute get_command_output tool
        output_tool = acp_agent._mock_tools["get_command_output"]  # type: ignore[attr-defined]
        result = await output_tool.execute(
            terminal_id="term_789",
            session_id="test_session",
        )

        # Verify result
        assert "Processing..." in result
        assert "[Still running]" in result

        acp_agent.connection.terminal_output.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_command_output_completed(self, acp_agent: LLMlingACPAgent):
        """Test getting output from completed command."""
        # Mock completed command
        acp_agent.connection.terminal_output = AsyncMock(  # type: ignore[method-assign]
            return_value=TerminalOutputResponse(
                output="Task completed\n",
                truncated=False,
                exit_status=TerminalExitStatus(exit_code=0, signal=None),
            )
        )

        # Execute get_command_output tool
        output_tool = acp_agent._mock_tools["get_command_output"]  # type: ignore[attr-defined]
        result = await output_tool.execute(
            terminal_id="term_completed",
            session_id="test_session",
        )

        # Verify result
        assert "Task completed" in result
        assert "[Exited with code 0]" in result

    @pytest.mark.asyncio
    async def test_get_command_output_truncated(self, acp_agent: LLMlingACPAgent):
        """Test handling of truncated output."""
        # Mock truncated output
        acp_agent.connection.terminal_output = AsyncMock(  # type: ignore[method-assign]
            return_value=TerminalOutputResponse(
                output="Large output...",
                truncated=True,
                exit_status=TerminalExitStatus(exit_code=0, signal=None),
            )
        )

        # Execute get_command_output tool
        output_tool = acp_agent._mock_tools["get_command_output"]  # type: ignore[attr-defined]
        result = await output_tool.execute(
            terminal_id="term_truncated",
            session_id="test_session",
        )

        # Verify truncation is indicated
        assert "Large output..." in result
        assert "[Output was truncated]" in result

    @pytest.mark.asyncio
    async def test_command_with_signal(self, acp_agent: LLMlingACPAgent):
        """Test command terminated by signal."""
        # Mock signal termination
        acp_agent.connection.terminal_output = AsyncMock(  # type: ignore[method-assign]
            return_value=TerminalOutputResponse(
                output="Interrupted\n",
                truncated=False,
                exit_status=TerminalExitStatus(exit_code=None, signal="SIGINT"),
            )
        )

        # Execute get_command_output tool
        output_tool = acp_agent._mock_tools["get_command_output"]  # type: ignore[attr-defined]
        result = await output_tool.execute(
            terminal_id="term_signal",
            session_id="test_session",
        )

        # Verify signal is indicated
        assert "Interrupted" in result
        assert "[Terminated by signal SIGINT]" in result

    @pytest.mark.asyncio
    async def test_command_with_environment_variables(self, acp_agent: LLMlingACPAgent):
        """Test command execution with environment variables."""
        # Mock successful command with env vars
        acp_agent.connection.create_terminal = AsyncMock(  # type: ignore[method-assign]
            return_value=CreateTerminalResponse(terminal_id="term_env")
        )
        acp_agent.connection.wait_for_terminal_exit = AsyncMock(  # type: ignore[method-assign]
            return_value=WaitForTerminalExitResponse(exit_code=0, signal=None)
        )
        acp_agent.connection.terminal_output = AsyncMock(  # type: ignore[method-assign]
            return_value=TerminalOutputResponse(
                output="FOO=bar\n",
                truncated=False,
                exit_status=TerminalExitStatus(exit_code=0, signal=None),
            )
        )
        acp_agent.connection.release_terminal = AsyncMock(  # type: ignore[method-assign]
            return_value=ReleaseTerminalResponse()
        )

        # Execute run_command tool with env vars
        run_tool = acp_agent._mock_tools["run_command"]  # type: ignore[attr-defined]
        result = await run_tool.execute(
            command="printenv",
            args=["FOO"],
            env={"FOO": "bar"},
            cwd="/tmp",
            session_id="test_session",
        )

        # Verify environment variables were passed
        assert "FOO=bar" in result

        # Check that create_terminal was called with correct parameters
        create_call = acp_agent.connection.create_terminal.call_args[0][0]
        assert create_call.command == "printenv"
        assert create_call.args == ["FOO"]
        assert create_call.cwd == "/tmp"
        assert len(create_call.env) == 1
        assert create_call.env[0].name == "FOO"
        assert create_call.env[0].value == "bar"

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, acp_agent: LLMlingACPAgent):
        """Test tool error handling when ACP operations fail."""
        # Mock ACP operation failure
        acp_agent.connection.create_terminal = AsyncMock(  # type: ignore[method-assign]
            side_effect=Exception("Connection failed")
        )

        # Execute run_command tool
        run_tool = acp_agent._mock_tools["run_command"]  # type: ignore[attr-defined]
        result = await run_tool.execute(
            command="echo",
            args=["test"],
            session_id="test_session",
        )

        # Verify error handling
        assert result.startswith("Error executing command:")
        assert "Connection failed" in result

    @pytest.mark.asyncio
    async def test_create_terminal_tool(self, acp_agent: LLMlingACPAgent):
        """Test create_terminal tool."""
        # Mock create terminal response
        acp_agent.connection.create_terminal = AsyncMock(  # type: ignore[method-assign]
            return_value=CreateTerminalResponse(terminal_id="term_abc123")
        )

        # Execute create_terminal tool
        create_tool = acp_agent._mock_tools["create_terminal"]  # type: ignore[attr-defined]
        result = await create_tool.execute(
            command="ls",
            args=["-la"],
            cwd="/home",
            session_id="test_session",
        )

        # Verify result contains terminal ID
        assert result == "term_abc123"
        acp_agent.connection.create_terminal.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_terminal_exit_tool(self, acp_agent: LLMlingACPAgent):
        """Test wait_for_terminal_exit tool."""
        # Mock wait for exit response
        acp_agent.connection.wait_for_terminal_exit = AsyncMock(  # type: ignore[method-assign]
            return_value=WaitForTerminalExitResponse(exit_code=0, signal=None)
        )

        # Execute wait_for_terminal_exit tool
        wait_tool = acp_agent._mock_tools["wait_for_terminal_exit"]  # type: ignore[attr-defined]
        result = await wait_tool.execute(
            terminal_id="term_xyz789",
            session_id="test_session",
        )

        # Verify result
        assert "term_xyz789 completed" in result
        assert "exit code 0" in result
        acp_agent.connection.wait_for_terminal_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_kill_terminal_tool(self, acp_agent: LLMlingACPAgent):
        """Test kill_terminal tool."""
        from acp.schema import KillTerminalCommandResponse

        # Mock kill terminal response
        acp_agent.connection.kill_terminal = AsyncMock(  # type: ignore[method-assign]
            return_value=KillTerminalCommandResponse()
        )

        # Execute kill_terminal tool
        kill_tool = acp_agent._mock_tools["kill_terminal"]  # type: ignore[attr-defined]
        result = await kill_tool.execute(
            terminal_id="term_kill123",
            session_id="test_session",
        )

        # Verify result
        assert "term_kill123 killed successfully" in result
        acp_agent.connection.kill_terminal.assert_called_once()

    @pytest.mark.asyncio
    async def test_release_terminal_tool(self, acp_agent: LLMlingACPAgent):
        """Test release_terminal tool."""
        from acp.schema import ReleaseTerminalResponse

        # Mock release terminal response
        acp_agent.connection.release_terminal = AsyncMock(  # type: ignore[method-assign]
            return_value=ReleaseTerminalResponse()
        )

        # Execute release_terminal tool
        release_tool = acp_agent._mock_tools["release_terminal"]  # type: ignore[attr-defined]
        result = await release_tool.execute(
            terminal_id="term_release456",
            session_id="test_session",
        )

        # Verify result
        assert "term_release456 released successfully" in result
        acp_agent.connection.release_terminal.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_command_with_timeout_success(self, acp_agent: LLMlingACPAgent):
        """Test run_command_with_timeout tool with successful completion."""
        # Mock all terminal operations
        acp_agent.connection.create_terminal = AsyncMock(  # type: ignore[method-assign]
            return_value=CreateTerminalResponse(terminal_id="term_timeout123")
        )
        acp_agent.connection.wait_for_terminal_exit = AsyncMock(  # type: ignore[method-assign]
            return_value=WaitForTerminalExitResponse(exit_code=0, signal=None)
        )
        acp_agent.connection.terminal_output = AsyncMock(  # type: ignore[method-assign]
            return_value=TerminalOutputResponse(
                output="Quick command output\n",
                truncated=False,
                exit_status=TerminalExitStatus(exit_code=0, signal=None),
            )
        )
        acp_agent.connection.release_terminal = AsyncMock(  # type: ignore[method-assign]
            return_value=ReleaseTerminalResponse()
        )

        # Execute run_command_with_timeout tool
        timeout_tool = acp_agent._mock_tools["run_command_with_timeout"]  # type: ignore[attr-defined]
        result = await timeout_tool.execute(
            command="echo",
            args=["quick"],
            timeout_seconds=5,
            session_id="test_session",
        )

        # Verify result
        assert "Quick command output" in result
        assert "[Command exited with code 0]" in result

        # Verify all operations were called
        acp_agent.connection.create_terminal.assert_called_once()
        acp_agent.connection.wait_for_terminal_exit.assert_called_once()
        acp_agent.connection.terminal_output.assert_called_once()
        acp_agent.connection.release_terminal.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_command_with_timeout_timeout(self, acp_agent: LLMlingACPAgent):
        """Test run_command_with_timeout tool with timeout."""
        import asyncio

        from acp.schema import KillTerminalCommandResponse

        # Mock create terminal
        acp_agent.connection.create_terminal = AsyncMock(  # type: ignore[method-assign]
            return_value=CreateTerminalResponse(terminal_id="term_slow123")
        )

        # Mock wait_for_terminal_exit to timeout
        async def slow_wait(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
            return WaitForTerminalExitResponse(exit_code=0, signal=None)

        acp_agent.connection.wait_for_terminal_exit = AsyncMock(  # type: ignore[method-assign]
            side_effect=slow_wait
        )

        # Mock kill and output responses
        acp_agent.connection.kill_terminal = AsyncMock(  # type: ignore[method-assign]
            return_value=KillTerminalCommandResponse()
        )
        acp_agent.connection.terminal_output = AsyncMock(  # type: ignore[method-assign]
            return_value=TerminalOutputResponse(
                output="Partial output...\n",
                truncated=False,
                exit_status=None,
            )
        )
        acp_agent.connection.release_terminal = AsyncMock(  # type: ignore[method-assign]
            return_value=ReleaseTerminalResponse()
        )

        # Execute run_command_with_timeout tool with short timeout
        timeout_tool = acp_agent._mock_tools["run_command_with_timeout"]  # type: ignore[attr-defined]
        result = await timeout_tool.execute(
            command="sleep",
            args=["100"],
            timeout_seconds=1,  # Short timeout
            session_id="test_session",
        )

        # Verify timeout handling
        assert "Partial output" in result
        assert "timed out after 1 seconds and was killed" in result

        # Verify kill was called
        acp_agent.connection.kill_terminal.assert_called_once()
        acp_agent.connection.release_terminal.assert_called_once()
