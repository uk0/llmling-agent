"""Tests for client capability checking in ACP agent initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from acp.schema import (
    ClientCapabilities,
    FileSystemCapability,
    InitializeRequest,
)


if TYPE_CHECKING:
    from llmling_agent_acp.acp_agent import LLMlingACPAgent


class TestClientCapabilities:
    """Test client capability checking and tool registration."""

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
        """Create ACP agent with no tools registered initially."""
        from llmling_agent_acp.acp_agent import LLMlingACPAgent

        # Create mock agent
        mock_agent = Mock()
        mock_tools = {}

        def register_tool(tool):
            mock_tools[tool.name] = tool

        mock_agent.tools = Mock()
        mock_agent.tools.register_tool = register_tool
        mock_agent_pool.agents = {"test_agent": mock_agent}

        # Create ACP agent
        agent = LLMlingACPAgent(
            connection=mock_connection,
            agent_pool=mock_agent_pool,
            terminal_access=True,  # Enable terminal access for testing
        )

        # Store reference to mock tools for testing
        agent._mock_tools = mock_tools  # type: ignore[attr-defined]
        return agent

    @pytest.mark.asyncio
    async def test_no_capabilities_no_tools(self, acp_agent: LLMlingACPAgent):
        """Test that no tools are registered when client has no capabilities."""
        # Initialize with no capabilities
        request = InitializeRequest(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=False, write_text_file=False),
                terminal=False,
            ),
        )

        await acp_agent.initialize(request)

        # No tools should be registered
        registered_tools = set(acp_agent._mock_tools.keys())  # type: ignore[attr-defined]
        assert len(registered_tools) == 0

    @pytest.mark.asyncio
    async def test_full_capabilities_all_tools(self, acp_agent: LLMlingACPAgent):
        """Test that all tools are registered when client supports everything."""
        # Initialize with full capabilities
        request = InitializeRequest(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=True, write_text_file=True),
                terminal=True,
            ),
        )

        await acp_agent.initialize(request)

        # All tools should be registered
        registered_tools = set(acp_agent._mock_tools.keys())  # type: ignore[attr-defined]

        expected_terminal_tools = {
            "run_command",
            "get_command_output",
            "create_terminal",
            "wait_for_terminal_exit",
            "kill_terminal",
            "release_terminal",
            "run_command_with_timeout",
        }
        expected_filesystem_tools = {"read_text_file", "write_text_file"}
        expected_all_tools = expected_terminal_tools | expected_filesystem_tools

        assert registered_tools == expected_all_tools

    @pytest.mark.asyncio
    async def test_terminal_only_capabilities(self, acp_agent: LLMlingACPAgent):
        """Test that only term tools are registered when only term is supported."""
        # Initialize with only terminal capabilities
        request = InitializeRequest(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=False, write_text_file=False),
                terminal=True,
            ),
        )

        await acp_agent.initialize(request)

        # Only terminal tools should be registered
        registered_tools = set(acp_agent._mock_tools.keys())  # type: ignore[attr-defined]

        expected_terminal_tools = {
            "run_command",
            "get_command_output",
            "create_terminal",
            "wait_for_terminal_exit",
            "kill_terminal",
            "release_terminal",
            "run_command_with_timeout",
        }

        assert registered_tools == expected_terminal_tools

        # Check that all registered tools are terminal tools
        for tool_name in registered_tools:
            tool = acp_agent._mock_tools[tool_name]  # type: ignore[attr-defined]
            assert tool.source == "terminal"

    @pytest.mark.asyncio
    async def test_filesystem_only_capabilities(self, acp_agent: LLMlingACPAgent):
        """Test that only fs tools are registered when only fs is supported."""
        # Initialize with only filesystem capabilities
        request = InitializeRequest(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=True, write_text_file=True),
                terminal=False,
            ),
        )

        await acp_agent.initialize(request)

        # Only filesystem tools should be registered
        registered_tools = set(acp_agent._mock_tools.keys())  # type: ignore[attr-defined]
        expected_filesystem_tools = {"read_text_file", "write_text_file"}

        assert registered_tools == expected_filesystem_tools

        # Check that all registered tools are filesystem tools
        for tool_name in registered_tools:
            tool = acp_agent._mock_tools[tool_name]  # type: ignore[attr-defined]
            assert tool.source == "filesystem"

    @pytest.mark.asyncio
    async def test_partial_filesystem_capabilities(self, acp_agent: LLMlingACPAgent):
        """Test that only supported filesystem tools are registered."""
        # Initialize with only read filesystem capability
        request = InitializeRequest(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=True, write_text_file=False),
                terminal=False,
            ),
        )

        await acp_agent.initialize(request)

        # Only read tool should be registered
        registered_tools = set(acp_agent._mock_tools.keys())  # type: ignore[attr-defined]
        assert registered_tools == {"read_text_file"}

        # Test with only write capability
        acp_agent._mock_tools.clear()  # type: ignore[attr-defined]

        request = InitializeRequest(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=False, write_text_file=True),
                terminal=False,
            ),
        )

        await acp_agent.initialize(request)

        # Only write tool should be registered
        registered_tools = set(acp_agent._mock_tools.keys())  # type: ignore[attr-defined]
        assert registered_tools == {"write_text_file"}

    @pytest.mark.asyncio
    async def test_terminal_disabled_locally(self, mock_connection, mock_agent_pool):
        """Test that terminal tools are not registered when terminal_access=False."""
        from llmling_agent_acp.acp_agent import LLMlingACPAgent

        # Create agent with terminal access disabled
        mock_agent = Mock()
        mock_tools = {}

        def register_tool(tool):
            mock_tools[tool.name] = tool

        mock_agent.tools = Mock()
        mock_agent.tools.register_tool = register_tool
        mock_agent_pool.agents = {"test_agent": mock_agent}

        agent = LLMlingACPAgent(
            connection=mock_connection,
            agent_pool=mock_agent_pool,
            terminal_access=False,  # Disabled locally
        )
        agent._mock_tools = mock_tools  # type: ignore[attr-defined]

        # Initialize with full capabilities
        request = InitializeRequest(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=True, write_text_file=True),
                terminal=True,  # Client supports terminal
            ),
        )

        await agent.initialize(request)

        # Only filesystem tools should be registered (terminal disabled locally)
        registered_tools = set(mock_tools.keys())
        expected_filesystem_tools = {"read_text_file", "write_text_file"}
        assert registered_tools == expected_filesystem_tools

    @pytest.mark.asyncio
    async def test_no_client_capabilities(self, acp_agent: LLMlingACPAgent):
        """Test handling when client_capabilities is None."""
        # Initialize with None capabilities
        request = InitializeRequest(
            protocol_version=1,
            client_capabilities=None,
        )

        await acp_agent.initialize(request)

        # No tools should be registered
        registered_tools = set(acp_agent._mock_tools.keys())  # type: ignore[attr-defined]
        assert len(registered_tools) == 0

    @pytest.mark.asyncio
    async def test_capabilities_stored_correctly(self, acp_agent: LLMlingACPAgent):
        """Test that client capabilities are stored correctly in the agent."""
        capabilities = ClientCapabilities(
            fs=FileSystemCapability(read_text_file=True, write_text_file=False),
            terminal=True,
        )

        request = InitializeRequest(
            protocol_version=1,
            client_capabilities=capabilities,
        )

        await acp_agent.initialize(request)

        # Capabilities should be stored
        assert acp_agent.client_capabilities
        assert acp_agent.client_capabilities.fs
        assert acp_agent.client_capabilities == capabilities
        assert acp_agent.client_capabilities.fs.read_text_file is True
        assert acp_agent.client_capabilities.fs.write_text_file is False
        assert acp_agent.client_capabilities.terminal is True

    @pytest.mark.asyncio
    async def test_protocol_version_negotiation(self, acp_agent: LLMlingACPAgent):
        """Test that protocol version is negotiated correctly."""
        # Test client with higher version
        request = InitializeRequest(
            protocol_version=10,  # Higher than agent's version
            client_capabilities=ClientCapabilities(),
        )

        response = await acp_agent.initialize(request)
        # Should use agent's maximum version
        assert response.protocol_version == acp_agent.PROTOCOL_VERSION

        # Test client with lower version
        request = InitializeRequest(
            protocol_version=1,  # Lower than agent's version
            client_capabilities=ClientCapabilities(),
        )

        response = await acp_agent.initialize(request)
        # Should use client's version
        assert response.protocol_version == 1
