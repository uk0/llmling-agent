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
from llmling_agent_acp.acp_tools import ACPCapabilityResourceProvider


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
        mock_agent_pool.agents = {"test_agent": mock_agent}

        return LLMlingACPAgent(
            connection=mock_connection,
            agent_pool=mock_agent_pool,
        )

    async def test_no_capabilities_no_tools(self, acp_agent: LLMlingACPAgent):
        """Test that no tools are registered when client has no capabilities."""
        # Test with ResourceProvider directly - no capabilities
        capabilities = ClientCapabilities(
            fs=FileSystemCapability(read_text_file=False, write_text_file=False),
            terminal=False,
        )

        provider = ACPCapabilityResourceProvider(
            agent=acp_agent,
            session_id="test_session",
            client_capabilities=capabilities,
        )

        tools = await provider.get_tools()
        assert len(tools) == 0

    async def test_full_capabilities(self, acp_agent: LLMlingACPAgent):
        """Test that all tools are registered when client has full capabilities."""
        # Test with ResourceProvider directly
        capabilities = ClientCapabilities(
            fs=FileSystemCapability(read_text_file=True, write_text_file=True),
            terminal=True,
        )

        provider = ACPCapabilityResourceProvider(
            agent=acp_agent,
            session_id="test_session",
            client_capabilities=capabilities,
        )

        tools = await provider.get_tools()
        tool_names = {tool.name for tool in tools}

        expected_tools = {
            # Terminal tools
            "run_command",
            "get_command_output",
            "create_terminal",
            "wait_for_terminal_exit",
            "kill_terminal",
            "release_terminal",
            "run_command_with_timeout",
            # Filesystem tools
            "read_text_file",
            "write_text_file",
        }

        assert tool_names == expected_tools

    async def test_terminal_only_capabilities(self, acp_agent: LLMlingACPAgent):
        """Test that only term tools are registered when only term is supported."""
        # Test with ResourceProvider directly
        capabilities = ClientCapabilities(
            fs=FileSystemCapability(read_text_file=False, write_text_file=False),
            terminal=True,
        )

        provider = ACPCapabilityResourceProvider(
            agent=acp_agent,
            session_id="test_session",
            client_capabilities=capabilities,
        )

        tools = await provider.get_tools()
        tool_names = {tool.name for tool in tools}

        expected_terminal_tools = {
            "run_command",
            "get_command_output",
            "create_terminal",
            "wait_for_terminal_exit",
            "kill_terminal",
            "release_terminal",
            "run_command_with_timeout",
        }

        assert tool_names == expected_terminal_tools

        # Check that all registered tools are terminal tools
        for tool in tools:
            assert tool.source == "terminal"

    async def test_filesystem_only_capabilities(self, acp_agent: LLMlingACPAgent):
        """Test that only fs tools are registered when only fs is supported."""
        # Test with ResourceProvider directly
        capabilities = ClientCapabilities(
            fs=FileSystemCapability(read_text_file=True, write_text_file=True),
            terminal=False,
        )

        provider = ACPCapabilityResourceProvider(
            agent=acp_agent,
            session_id="test_session",
            client_capabilities=capabilities,
        )

        tools = await provider.get_tools()
        tool_names = {tool.name for tool in tools}
        expected_filesystem_tools = {"read_text_file", "write_text_file"}

        assert tool_names == expected_filesystem_tools

        # Check that all registered tools are filesystem tools
        for tool in tools:
            assert tool.source == "filesystem"

    async def test_partial_filesystem_capabilities_read_only(
        self, acp_agent: LLMlingACPAgent
    ):
        """Test that only read tool is registered when only read is enabled."""
        # Test with ResourceProvider directly - read only
        capabilities = ClientCapabilities(
            fs=FileSystemCapability(read_text_file=True, write_text_file=False),
            terminal=False,
        )

        provider = ACPCapabilityResourceProvider(
            agent=acp_agent,
            session_id="test_session",
            client_capabilities=capabilities,
        )

        tools = await provider.get_tools()
        tool_names = {tool.name for tool in tools}
        assert tool_names == {"read_text_file"}

    async def test_partial_filesystem_capabilities_write_only(
        self, acp_agent: LLMlingACPAgent
    ):
        """Test that only write tool is registered when only write is enabled."""
        # Test with ResourceProvider directly - write only
        capabilities = ClientCapabilities(
            fs=FileSystemCapability(read_text_file=False, write_text_file=True),
            terminal=False,
        )

        provider = ACPCapabilityResourceProvider(
            agent=acp_agent,
            session_id="test_session",
            client_capabilities=capabilities,
        )

        tools = await provider.get_tools()
        tool_names = {tool.name for tool in tools}
        assert tool_names == {"write_text_file"}

    async def test_mixed_capabilities_with_terminal_disabled_locally(
        self, mock_connection, mock_agent_pool
    ):
        """Test mixed capabilities with terminal disabled locally."""
        from llmling_agent_acp.acp_agent import LLMlingACPAgent

        # Create agent with terminal access disabled
        mock_agent = Mock()
        mock_agent_pool.agents = {"test_agent": mock_agent}

        agent = LLMlingACPAgent(
            connection=mock_connection,
            agent_pool=mock_agent_pool,
            terminal_access=False,  # Disabled locally
        )

        # Test with ResourceProvider directly
        capabilities = ClientCapabilities(
            fs=FileSystemCapability(read_text_file=True, write_text_file=True),
            terminal=True,  # Client supports terminal
        )

        provider = ACPCapabilityResourceProvider(
            agent=agent,
            session_id="test_session",
            client_capabilities=capabilities,
        )

        tools = await provider.get_tools()
        tool_names = {tool.name for tool in tools}
        expected_filesystem_tools = {"read_text_file", "write_text_file"}
        assert tool_names == expected_filesystem_tools

    async def test_none_client_capabilities(self, acp_agent: LLMlingACPAgent):
        """Test handling when client_capabilities is None."""
        # Initialize with None capabilities
        request = InitializeRequest(
            protocol_version=1,
            client_capabilities=None,
        )

        await acp_agent.initialize(request)

        # Initialize doesn't register tools anymore - tools are registered via
        # ResourceProvider
        # This test verifies that initialize works without error with None caps
        assert acp_agent.client_capabilities is None

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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
