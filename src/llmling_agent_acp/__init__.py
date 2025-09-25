"""ACP (Agent Client Protocol) integration for llmling-agent."""

from __future__ import annotations

from acp import DefaultACPClient
from llmling_agent_acp.server import ACPServer
from llmling_agent_acp.acp_agent import LLMlingACPAgent
from llmling_agent_acp.wrappers import ACPClientInterface
from llmling_agent_acp.session import ACPSession, ACPSessionManager
from llmling_agent_acp.command_bridge import ACPCommandBridge
from llmling_agent_acp.converters import (
    FileSystemBridge,
    convert_acp_mcp_server_to_config,
    from_content_blocks,
    to_content_blocks,
    to_session_updates,
)
from llmling_agent_acp.tools import ACPToolBridge
from acp.acp_types import StopReason


__all__ = [
    "ACPClientInterface",
    "ACPCommandBridge",
    "ACPServer",
    "ACPSession",
    "ACPSessionManager",
    "ACPToolBridge",
    "DefaultACPClient",
    "FileSystemBridge",
    "LLMlingACPAgent",
    "StopReason",
    "convert_acp_mcp_server_to_config",
    "from_content_blocks",
    "to_content_blocks",
    "to_session_updates",
]
