"""Tests for session-scoped ACP capability provider and tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

from pydantic_ai import RunContext, RunUsage
from pydantic_ai.models.test import TestModel
import pytest

from acp import AgentSideConnection
from acp.schema import (
    ClientCapabilities,
    CreateTerminalResponse,
    FileSystemCapability,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    TerminalExitStatus,
    TerminalOutputResponse,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)
from llmling_agent import AgentPool
from llmling_agent_acp.acp_agent import LLMlingACPAgent
from llmling_agent_acp.acp_tools import ACPCapabilityResourceProvider


CTX = RunContext(tool_call_id="test", deps=None, model=TestModel(), usage=RunUsage())


class TestACPCapabilityProvider:
    """Test ACP capability resource provider for session-scoped tools."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock ACP connection."""
        return Mock(spec=AgentSideConnection)

    @pytest.fixture
    def mock_agent_pool(self):
        """Create a mock agent pool with a single agent."""
        # Create mock agent
        mock_agent = Mock()
        mock_agent.name = "test_agent"
        mock_agent.description = "Test agent"
        mock_agent.tools = Mock()
        mock_agent.tools.add_provider = Mock()
        mock_agent.tools.remove_provider = Mock()

        # Create mock pool
        pool = Mock(spec=AgentPool)
        pool.agents = {"test_agent": mock_agent}
        pool.get_agent = Mock(return_value=mock_agent)

        return pool

    @pytest.fixture
    def acp_agent(self, mock_connection, mock_agent_pool):
        """Create ACP agent."""
        return LLMlingACPAgent(
            connection=mock_connection,
            agent_pool=mock_agent_pool,
            terminal_access=True,
            file_access=True,
        )


class TestSessionScopedTerminalTools:
    """Test terminal tools with session-scoped functionality."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock ACP connection with terminal responses."""
        connection = Mock(spec=AgentSideConnection)

        # Mock terminal operations
        connection.create_terminal = AsyncMock(
            return_value=CreateTerminalResponse(terminal_id="term_123")
        )
        connection.wait_for_terminal_exit = AsyncMock(
            return_value=WaitForTerminalExitResponse(exit_code=0)
        )
        connection.terminal_output = AsyncMock(
            return_value=TerminalOutputResponse(
                output="Hello World\n",
                truncated=False,
                exit_status=TerminalExitStatus(exit_code=0),
            )
        )
        connection.release_terminal = AsyncMock(return_value=ReleaseTerminalResponse())
        connection.kill_terminal = AsyncMock()

        # Mock filesystem operations
        connection.read_text_file = AsyncMock(
            return_value=ReadTextFileResponse(content="file content")
        )
        connection.write_text_file = AsyncMock(return_value=WriteTextFileResponse())

        return connection

    @pytest.fixture
    def acp_agent(self, mock_connection):
        """Create ACP agent with mocked connection."""
        # Create mock agent pool
        mock_agent = Mock()
        mock_agent.name = "test_agent"
        pool = Mock(spec=AgentPool)
        pool.agents = {"test_agent": mock_agent}

        return LLMlingACPAgent(
            connection=mock_connection,
            agent_pool=pool,
            terminal_access=True,
            file_access=True,
        )

    @pytest.fixture
    async def provider_with_tools(self, acp_agent):
        """Create provider with full capabilities and get tools."""
        capabilities = ClientCapabilities(
            fs=FileSystemCapability(read_text_file=True, write_text_file=True),
            terminal=True,
        )

        provider = ACPCapabilityResourceProvider(
            agent=acp_agent,
            session_id="test_session_123",
            client_capabilities=capabilities,
        )

        tools = await provider.get_tools()
        tool_dict = {tool.name: tool for tool in tools}

        return provider, tool_dict

    async def test_run_command_success(self, provider_with_tools):
        """Test run_command tool executes successfully."""
        provider, tools = provider_with_tools
        run_tool = tools["run_command"]

        result = await run_tool.execute(ctx=CTX, command="echo", args=["Hello World"])

        assert "Hello World" in result
        assert result == "Command completed with exit code 0:\nOutput:\nHello World\n"

        # Verify ACP calls were made with correct session_id
        agent = provider.agent
        agent.connection.create_terminal.assert_called_once()
        create_call = agent.connection.create_terminal.call_args[0][0]
        assert create_call.session_id == "test_session_123"
        assert create_call.command == "echo"
        assert create_call.args == ["Hello World"]

    async def test_run_command_with_environment(self, provider_with_tools):
        """Test run_command with environment variables."""
        provider, tools = provider_with_tools
        run_tool = tools["run_command"]

        await run_tool.execute(
            ctx=CTX, command="env", env={"TEST_VAR": "test_value", "FOO": "bar"}
        )

        # Verify environment variables were passed
        agent = provider.agent
        create_call = agent.connection.create_terminal.call_args[0][0]
        assert len(create_call.env) == 2  # noqa: PLR2004
        env_dict = {var.name: var.value for var in create_call.env}
        assert env_dict == {"TEST_VAR": "test_value", "FOO": "bar"}

    async def test_create_terminal_tool(self, provider_with_tools):
        """Test create_terminal tool returns terminal ID."""
        provider, tools = provider_with_tools
        create_tool = tools["create_terminal"]

        result = await create_tool.execute(ctx=CTX, command="ls", args=["-la"])

        assert result == "term_123"

        # Verify session_id was used
        agent = provider.agent
        create_call = agent.connection.create_terminal.call_args[0][0]
        assert create_call.session_id == "test_session_123"

    async def test_get_command_output_tool(self, provider_with_tools):
        """Test get_command_output tool retrieves output."""
        provider, tools = provider_with_tools
        output_tool = tools["get_command_output"]

        result = await output_tool.execute(ctx=CTX, terminal_id="term_123")

        assert "Hello World" in result
        assert "[Exited with code 0]" in result

        # Verify session_id was used
        agent = provider.agent
        output_call = agent.connection.terminal_output.call_args[0][0]
        assert output_call.session_id == "test_session_123"
        assert output_call.terminal_id == "term_123"

    async def test_wait_for_terminal_exit_tool(self, provider_with_tools):
        """Test wait_for_terminal_exit tool."""
        provider, tools = provider_with_tools
        wait_tool = tools["wait_for_terminal_exit"]

        result = await wait_tool.execute(ctx=CTX, terminal_id="term_123")

        assert "Terminal term_123 completed with exit code 0" in result

        # Verify session_id was used
        agent = provider.agent
        wait_call = agent.connection.wait_for_terminal_exit.call_args[0][0]
        assert wait_call.session_id == "test_session_123"

    async def test_kill_terminal_tool(self, provider_with_tools):
        """Test kill_terminal tool."""
        provider, tools = provider_with_tools
        kill_tool = tools["kill_terminal"]

        result = await kill_tool.execute(ctx=CTX, terminal_id="term_123")

        assert "Terminal term_123 killed successfully" in result

        # Verify session_id was used
        agent = provider.agent
        kill_call = agent.connection.kill_terminal.call_args[0][0]
        assert kill_call.session_id == "test_session_123"

    async def test_release_terminal_tool(self, provider_with_tools):
        """Test release_terminal tool."""
        provider, tools = provider_with_tools
        release_tool = tools["release_terminal"]

        result = await release_tool.execute(ctx=CTX, terminal_id="term_123")

        assert "Terminal term_123 released successfully" in result

        # Verify session_id was used
        agent = provider.agent
        release_call = agent.connection.release_terminal.call_args[0][0]
        assert release_call.session_id == "test_session_123"

    async def test_run_command_with_timeout_success(self, provider_with_tools):
        """Test run_command_with_timeout completes successfully."""
        _provider, tools = provider_with_tools
        timeout_tool = tools["run_command"]

        result = await timeout_tool.execute(
            ctx=CTX, command="echo", args=["test"], timeout_seconds=5
        )

        assert "Hello World" in result
        assert result == "Command completed with exit code 0:\nOutput:\nHello World\n"

    async def test_terminal_error_handling(self, provider_with_tools):
        """Test terminal tool error handling."""
        provider, tools = provider_with_tools

        # Make connection raise an error
        provider.agent.connection.create_terminal.side_effect = Exception(
            "Connection error"
        )

        run_tool = tools["run_command"]
        result = await run_tool.execute(ctx=CTX, command="echo", args=["test"])

        assert "Error executing command: Connection error" in result


class TestSessionScopedFilesystemTools:
    """Test filesystem tools with session-scoped functionality."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock ACP connection with filesystem responses."""
        connection = Mock(spec=AgentSideConnection)

        connection.read_text_file = AsyncMock(
            return_value=ReadTextFileResponse(content="file contents here")
        )
        connection.write_text_file = AsyncMock(return_value=WriteTextFileResponse())

        return connection

    @pytest.fixture
    def acp_agent(self, mock_connection):
        """Create ACP agent with mocked connection."""
        mock_agent = Mock()
        mock_agent.name = "test_agent"
        pool = Mock(spec=AgentPool)
        pool.agents = {"test_agent": mock_agent}

        return LLMlingACPAgent(
            connection=mock_connection,
            agent_pool=pool,
            file_access=True,
        )

    @pytest.fixture
    async def provider_with_fs_tools(self, acp_agent):
        """Create provider with filesystem capabilities and get tools."""
        capabilities = ClientCapabilities(
            fs=FileSystemCapability(read_text_file=True, write_text_file=True),
        )

        provider = ACPCapabilityResourceProvider(
            agent=acp_agent,
            session_id="fs_session_456",
            client_capabilities=capabilities,
        )

        tools = await provider.get_tools()
        tool_dict = {tool.name: tool for tool in tools}

        return provider, tool_dict

    async def test_read_text_file_tool(self, provider_with_fs_tools):
        """Test read_text_file tool."""
        provider, tools = provider_with_fs_tools
        read_tool = tools["read_text_file"]

        result = await read_tool.execute(ctx=CTX, path="/test/file.txt")

        assert result == "file contents here"

        # Verify session_id was used
        agent = provider.agent
        read_call = agent.connection.read_text_file.call_args[0][0]
        assert read_call.session_id == "fs_session_456"
        assert read_call.path == "/test/file.txt"

    async def test_read_text_file_with_line_limit(self, provider_with_fs_tools):
        """Test read_text_file with line and limit parameters."""
        provider, tools = provider_with_fs_tools
        read_tool = tools["read_text_file"]

        await read_tool.execute(ctx=CTX, path="/test/file.txt", line=10, limit=5)

        # Verify parameters were passed
        agent = provider.agent
        read_call = agent.connection.read_text_file.call_args[0][0]
        assert read_call.line == 10  # noqa: PLR2004
        assert read_call.limit == 5  # noqa: PLR2004

    async def test_write_text_file_tool(self, provider_with_fs_tools):
        """Test write_text_file tool."""
        provider, tools = provider_with_fs_tools
        write_tool = tools["write_text_file"]

        result = await write_tool.execute(
            ctx=CTX, path="/test/output.txt", content="Hello, World!"
        )

        assert "Successfully wrote file: /test/output.txt" in result

        # Verify session_id was used
        agent = provider.agent
        write_call = agent.connection.write_text_file.call_args[0][0]
        assert write_call.session_id == "fs_session_456"
        assert write_call.path == "/test/output.txt"
        assert write_call.content == "Hello, World!"

    async def test_filesystem_error_handling(self, provider_with_fs_tools):
        """Test filesystem tool error handling."""
        provider, tools = provider_with_fs_tools

        # Make connection raise an error
        provider.agent.connection.read_text_file.side_effect = Exception("File not found")

        read_tool = tools["read_text_file"]
        result = await read_tool.execute(ctx=CTX, path="/nonexistent.txt")

        assert "Error reading file: File not found" in result


class TestAgentSwitchingWithCapabilityProvider:
    """Test capability provider movement during agent switching."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock ACP connection."""
        connection = Mock(spec=AgentSideConnection)
        connection.create_terminal = AsyncMock(
            return_value=CreateTerminalResponse(terminal_id="term_switch")
        )
        return connection

    @pytest.fixture
    def multi_agent_pool(self):
        """Create a mock agent pool with multiple agents."""
        # Create multiple mock agents
        agent_a = Mock()
        agent_a.name = "agent_a"
        agent_a.tools = Mock()
        agent_a.tools.add_provider = Mock()
        agent_a.tools.remove_provider = Mock()

        agent_b = Mock()
        agent_b.name = "agent_b"
        agent_b.tools = Mock()
        agent_b.tools.add_provider = Mock()
        agent_b.tools.remove_provider = Mock()

        pool = Mock(spec=AgentPool)
        pool.agents = {"agent_a": agent_a, "agent_b": agent_b}
        pool.get_agent = Mock(side_effect=lambda name: pool.agents[name])

        return pool

    @pytest.fixture
    async def session_with_multiple_agents(self, mock_connection, multi_agent_pool):
        """Create a session with multiple agents and capability provider."""
        from llmling_agent_acp.session import ACPSession

        capabilities = ClientCapabilities(terminal=True)

        # Create ACP agent
        acp_agent = LLMlingACPAgent(
            connection=mock_connection,
            agent_pool=multi_agent_pool,
            terminal_access=True,
        )

        # Create session (starts with agent_a as current)
        session = ACPSession(
            session_id="multi_agent_session",
            agent_pool=multi_agent_pool,
            current_agent_name="agent_a",
            cwd="/tmp",
            client=Mock(),
            acp_agent=acp_agent,
            client_capabilities=capabilities,
        )

        return session, multi_agent_pool

    async def test_capability_provider_moves_on_agent_switch(
        self, session_with_multiple_agents
    ):
        """Test that capability provider moves when switching agents."""
        session, agent_pool = session_with_multiple_agents

        # Verify initial state - provider should be on agent_a
        agent_a = agent_pool.agents["agent_a"]
        agent_b = agent_pool.agents["agent_b"]

        # Provider should be added to agent_a initially
        assert agent_a.tools.add_provider.call_count == 1
        assert agent_b.tools.add_provider.call_count == 0

        # Get the provider that was added
        initial_provider = agent_a.tools.add_provider.call_args[0][0]
        assert initial_provider is not None

        # Switch to agent_b
        await session.switch_active_agent("agent_b")

        # Verify provider was removed from agent_a and added to agent_b
        agent_a.tools.remove_provider.assert_called_once_with(initial_provider)
        agent_b.tools.add_provider.assert_called_once_with(initial_provider)

        # Verify the same provider instance was moved (not recreated)
        moved_provider = agent_b.tools.add_provider.call_args[0][0]
        assert moved_provider is initial_provider

    async def test_current_agent_name_updates_on_switch(
        self, session_with_multiple_agents
    ):
        """Test that current agent name updates correctly."""
        session, _agent_pool = session_with_multiple_agents

        # Initial state
        assert session.current_agent_name == "agent_a"

        # Switch to agent_b
        await session.switch_active_agent("agent_b")

        # Verify current agent changed
        assert session.current_agent_name == "agent_b"

    async def test_switch_to_nonexistent_agent_raises_error(
        self, session_with_multiple_agents
    ):
        """Test that switching to non-existent agent raises ValueError."""
        session, _agent_pool = session_with_multiple_agents

        with pytest.raises(ValueError, match="Agent 'nonexistent' not found"):
            await session.switch_active_agent("nonexistent")

        # Verify current agent unchanged
        assert session.current_agent_name == "agent_a"

    async def test_no_provider_movement_when_no_capability_provider(
        self, multi_agent_pool
    ):
        """Test agent switching works when no capability provider exists."""
        from llmling_agent_acp.session import ACPSession

        # Create session without capability provider
        session = ACPSession(
            session_id="no_provider_session",
            agent_pool=multi_agent_pool,
            current_agent_name="agent_a",
            cwd="/tmp",
            client=Mock(),
            # No acp_agent or client_capabilities = no capability provider
        )

        # Should be able to switch without errors
        await session.switch_active_agent("agent_b")

        # Verify current agent changed
        assert session.current_agent_name == "agent_b"

        # Verify no provider operations were attempted
        agent_a = multi_agent_pool.agents["agent_a"]
        agent_b = multi_agent_pool.agents["agent_b"]

        agent_a.tools.remove_provider.assert_not_called()
        agent_b.tools.add_provider.assert_not_called()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
