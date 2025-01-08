"""MCP tool integration for LLMling agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling import LLMCallableTool

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp.types import Tool as MCPTool
    from pydantic_ai import RunContext

    from llmling_agent.mcp_server.client import MCPClient
    from llmling_agent.models.context import AgentContext
    from llmling_agent.tools.base import ToolInfo
    from llmling_agent.tools.manager import ToolManager


logger = get_logger(__name__)


def create_mcp_tool_callable(
    mcp_client: MCPClient,
    tool: MCPTool,
) -> Callable[[RunContext[AgentContext]], Any]:
    """Create a callable that forwards to MCP tool."""

    async def call_mcp_tool(ctx: RunContext[AgentContext], **kwargs: Any) -> str:
        """Forward call to MCP server."""
        return await mcp_client.call_tool(tool.name, kwargs)

    # Set metadata for LLMCallableTool creation
    call_mcp_tool.__name__ = tool.name
    call_mcp_tool.__doc__ = tool.description

    return call_mcp_tool


def register_mcp_tools(
    tool_manager: ToolManager,
    mcp_client: MCPClient,
) -> list[ToolInfo]:
    """Register MCP tools with tool manager.

    Args:
        tool_manager: Tool manager to register with
        mcp_client: Connected MCP client

    Returns:
        List of registered tool infos
    """
    registered = []

    for mcp_tool in mcp_client._available_tools:
        # Create callable that forwards to MCP
        tool_callable = create_mcp_tool_callable(mcp_client, mcp_tool)

        # Create LLMling tool
        llm_tool = LLMCallableTool.from_callable(
            tool_callable,
            schema_override={
                "name": mcp_tool.name,
                "description": mcp_tool.description or "No description",
                "parameters": mcp_tool.inputSchema,  # type: ignore
            },
        )

        # Register with manager
        metadata = {"mcp_tool": mcp_tool.name}
        tool_info = tool_manager.register_tool(llm_tool, source="mcp", metadata=metadata)
        registered.append(tool_info)

        logger.debug("Registered MCP tool: %s", mcp_tool.name)

    return registered
