"""MCP tool integration for LLMling agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling import LLMCallableTool

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.mcp_server.client import MCPClient
    from llmling_agent.tools.base import ToolInfo
    from llmling_agent.tools.manager import ToolManager


logger = get_logger(__name__)


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
    from llmling_agent.models.context import create_mcp_tool_callable

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
