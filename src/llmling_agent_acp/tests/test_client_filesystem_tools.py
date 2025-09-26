"""Tests for client-side filesystem tools that make ACP requests."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from acp.schema import (
    ClientCapabilities,
    FileSystemCapability,
    InitializeRequest,
    ReadTextFileResponse,
    WriteTextFileResponse,
)
from llmling_agent_acp.acp_agent import LLMlingACPAgent
from llmling_agent_acp.resource_providers import ACPCapabilityResourceProvider


class TestClientFilesystemTools:
    """Test client-side filesystem tools that request file operations from ACP client."""

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
    async def acp_agent(self, mock_connection, mock_agent_pool):
        """Create ACP agent with filesystem support."""
        # Create mock agent
        mock_agent = Mock()
        mock_tools = {}

        def register_tool(tool):
            mock_tools[tool.name] = tool

        mock_agent.tools = Mock()
        mock_agent.tools.register_tool = register_tool
        mock_agent_pool.agents = {"test_agent": mock_agent}

        # Create ACP agent (filesystem tools are always registered)
        agent = LLMlingACPAgent(
            connection=mock_connection,
            agent_pool=mock_agent_pool,
            terminal_access=False,  # Disable terminal tools for cleaner testing
        )

        # Initialize with filesystem capabilities
        await agent.initialize(
            InitializeRequest(
                protocol_version=1,
                client_capabilities=ClientCapabilities(
                    fs=FileSystemCapability(read_text_file=True, write_text_file=True),
                    terminal=False,
                ),
            )
        )

        return agent

    @pytest.fixture
    async def fs_provider(self, acp_agent):
        """Create filesystem capability provider for testing."""
        capabilities = ClientCapabilities(
            fs=FileSystemCapability(read_text_file=True, write_text_file=True),
            terminal=False,
        )

        return ACPCapabilityResourceProvider(
            agent=acp_agent,
            session_id="test_session",
            client_capabilities=capabilities,
        )

    @pytest.mark.asyncio
    async def test_filesystem_tools_registered(self, fs_provider):
        """Test that filesystem tools are registered."""
        tools = await fs_provider.get_tools()
        tool_names = {tool.name for tool in tools}

        expected_tools = {"read_text_file", "write_text_file"}
        assert expected_tools.issubset(tool_names)

        # Check that filesystem tools have correct source
        fs_tools = [tool for tool in tools if tool.name in expected_tools]
        for tool in fs_tools:
            assert tool.source == "filesystem"

    @pytest.mark.asyncio
    async def test_read_text_file_success(self, acp_agent: LLMlingACPAgent, fs_provider):
        """Test successful file reading."""
        # Mock read file response
        acp_agent.connection.read_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=ReadTextFileResponse(
                content="Hello, World!\nThis is a test file.\n"
            )
        )

        # Get read_text_file tool from provider
        tools = await fs_provider.get_tools()
        read_tool = next(tool for tool in tools if tool.name == "read_text_file")
        result = await read_tool.execute(
            path="/home/user/test.txt",
            session_id="test_session",
        )

        # Verify result
        assert "Hello, World!" in result
        assert "This is a test file." in result

        # Verify ACP call was made
        acp_agent.connection.read_text_file.assert_called_once()
        call_args = acp_agent.connection.read_text_file.call_args[0][0]
        assert call_args.path == "/home/user/test.txt"
        assert call_args.session_id == "test_session"

    @pytest.mark.asyncio
    async def test_read_text_file_with_line_and_limit(
        self, acp_agent: LLMlingACPAgent, fs_provider
    ):
        """Test file reading with line and limit parameters."""
        # Mock read file response
        acp_agent.connection.read_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=ReadTextFileResponse(content="Line 10\nLine 11\nLine 12\n")
        )

        # Get read_text_file tool from provider
        tools = await fs_provider.get_tools()
        read_tool = next(tool for tool in tools if tool.name == "read_text_file")
        result = await read_tool.execute(
            path="/home/user/large_file.txt",
            line=10,
            limit=3,
            session_id="test_session",
        )

        # Verify result
        assert "Line 10" in result
        assert "Line 11" in result
        assert "Line 12" in result

        # Verify ACP call parameters
        call_args = acp_agent.connection.read_text_file.call_args[0][0]
        assert call_args.path == "/home/user/large_file.txt"
        assert call_args.line == 10  # noqa: PLR2004
        assert call_args.limit == 3  # noqa: PLR2004

    @pytest.mark.asyncio
    async def test_read_text_file_error(self, acp_agent: LLMlingACPAgent, fs_provider):
        """Test file reading error handling."""
        # Mock read file error
        acp_agent.connection.read_text_file = AsyncMock(  # type: ignore[method-assign]
            side_effect=FileNotFoundError("File not found")
        )

        # Get read_text_file tool from provider
        tools = await fs_provider.get_tools()
        read_tool = next(tool for tool in tools if tool.name == "read_text_file")

        result = await read_tool.execute(
            path="/home/user/nonexistent.txt",
            session_id="test_session",
        )

        # Verify error handling
        assert result.startswith("Error reading file:")
        assert "File not found" in result

    @pytest.mark.asyncio
    async def test_write_text_file_success(self, acp_agent: LLMlingACPAgent, fs_provider):
        """Test successful file writing."""
        # Mock write file response
        acp_agent.connection.write_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=WriteTextFileResponse()
        )

        # Get write_text_file tool from provider
        tools = await fs_provider.get_tools()
        write_tool = next(tool for tool in tools if tool.name == "write_text_file")
        result = await write_tool.execute(
            path="/home/user/output.txt",
            content="Hello, World!\nThis is written content.\n",
            session_id="test_session",
        )

        # Verify result
        assert "Successfully wrote file" in result
        assert "/home/user/output.txt" in result

        # Verify ACP call was made
        acp_agent.connection.write_text_file.assert_called_once()
        call_args = acp_agent.connection.write_text_file.call_args[0][0]
        assert call_args.path == "/home/user/output.txt"
        assert call_args.content == "Hello, World!\nThis is written content.\n"
        assert call_args.session_id == "test_session"

    @pytest.mark.asyncio
    async def test_write_text_file_json(self, acp_agent: LLMlingACPAgent):
        """Test writing JSON content."""
        # Mock write file response
        acp_agent.connection.write_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=WriteTextFileResponse()
        )

        json_content = '{\n  "debug": true,\n  "version": "1.0.0"\n}'

        # Execute write_text_file tool
        write_tool = acp_agent._mock_tools["write_text_file"]  # type: ignore[attr-defined]
        result = await write_tool.execute(
            path="/home/user/config.json",
            content=json_content,
            session_id="test_session",
        )

        # Verify result
        assert "Successfully wrote file" in result
        assert "/home/user/config.json" in result

        # Verify content was written correctly
        call_args = acp_agent.connection.write_text_file.call_args[0][0]
        assert call_args.content == json_content

    @pytest.mark.asyncio
    async def test_write_text_file_error(self, acp_agent: LLMlingACPAgent, fs_provider):
        """Test file writing error handling."""
        # Mock write file error
        acp_agent.connection.write_text_file = AsyncMock(  # type: ignore[method-assign]
            side_effect=PermissionError("Permission denied")
        )

        # Get write_text_file tool from provider
        tools = await fs_provider.get_tools()
        write_tool = next(tool for tool in tools if tool.name == "write_text_file")
        result = await write_tool.execute(
            path="/root/protected.txt",
            content="This should fail",
            session_id="test_session",
        )

        # Verify error handling
        assert result.startswith("Error writing file:")
        assert "Permission denied" in result

    @pytest.mark.asyncio
    async def test_read_empty_file(self, acp_agent: LLMlingACPAgent):
        """Test reading an empty file."""
        # Mock empty file response
        acp_agent.connection.read_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=ReadTextFileResponse(content="")
        )

        # Execute read_text_file tool
        read_tool = acp_agent._mock_tools["read_text_file"]  # type: ignore[attr-defined]
        result = await read_tool.execute(
            path="/home/user/empty.txt",
            session_id="test_session",
        )

        # Verify empty content is handled correctly
        assert result == ""

    @pytest.mark.asyncio
    async def test_write_empty_file(self, acp_agent: LLMlingACPAgent):
        """Test writing empty content to a file."""
        # Mock write file response
        acp_agent.connection.write_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=WriteTextFileResponse()
        )

        # Execute write_text_file tool with empty content
        write_tool = acp_agent._mock_tools["write_text_file"]  # type: ignore[attr-defined]
        result = await write_tool.execute(
            path="/home/user/empty_output.txt",
            content="",
            session_id="test_session",
        )

        # Verify result
        assert "Successfully wrote file" in result

        # Verify empty content was written
        call_args = acp_agent.connection.write_text_file.call_args[0][0]
        assert call_args.content == ""

    @pytest.mark.asyncio
    async def test_read_file_with_unicode(self, acp_agent: LLMlingACPAgent):
        """Test reading file with unicode content."""
        unicode_content = "Hello 世界! 🌍\nThis has émojis and spëcial chars: café"

        # Mock read file response with unicode
        acp_agent.connection.read_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=ReadTextFileResponse(content=unicode_content)
        )

        # Execute read_text_file tool
        read_tool = acp_agent._mock_tools["read_text_file"]  # type: ignore[attr-defined]
        result = await read_tool.execute(
            path="/home/user/unicode.txt",
            session_id="test_session",
        )

        # Verify unicode content is preserved
        assert "世界" in result
        assert "🌍" in result
        assert "émojis" in result
        assert "café" in result

    @pytest.mark.asyncio
    async def test_write_file_with_unicode(self, acp_agent: LLMlingACPAgent):
        """Test writing file with unicode content."""
        unicode_content = "Testing unicode: 日本語, русский, العربية 🎉"

        # Mock write file response
        acp_agent.connection.write_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=WriteTextFileResponse()
        )

        # Execute write_text_file tool
        write_tool = acp_agent._mock_tools["write_text_file"]  # type: ignore[attr-defined]
        result = await write_tool.execute(
            path="/home/user/unicode_output.txt",
            content=unicode_content,
            session_id="test_session",
        )

        # Verify result
        assert "Successfully wrote file" in result

        # Verify unicode content was written correctly
        call_args = acp_agent.connection.write_text_file.call_args[0][0]
        assert call_args.content == unicode_content

    @pytest.mark.asyncio
    async def test_file_operations_with_different_sessions(
        self, acp_agent: LLMlingACPAgent
    ):
        """Test file operations with different session IDs."""
        # Mock responses
        acp_agent.connection.read_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=ReadTextFileResponse(content="session content")
        )
        acp_agent.connection.write_text_file = AsyncMock(  # type: ignore[method-assign]
            return_value=WriteTextFileResponse()
        )

        # Test with custom session ID
        read_tool = acp_agent._mock_tools["read_text_file"]  # type: ignore[attr-defined]
        await read_tool.execute(
            path="/home/user/test.txt",
            session_id="custom_session_123",
        )

        write_tool = acp_agent._mock_tools["write_text_file"]  # type: ignore[attr-defined]
        await write_tool.execute(
            path="/home/user/test.txt",
            content="test content",
            session_id="custom_session_456",
        )

        # Verify session IDs were passed correctly
        read_call_args = acp_agent.connection.read_text_file.call_args[0][0]
        assert read_call_args.session_id == "custom_session_123"

        write_call_args = acp_agent.connection.write_text_file.call_args[0][0]
        assert write_call_args.session_id == "custom_session_456"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
