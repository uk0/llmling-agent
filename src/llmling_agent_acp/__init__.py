"""ACP (Agent Client Protocol) integration for llmling-agent.

This package provides seamless integration between llmling-agent and
the Agent Client Protocol,
using the external acp library for robust JSON-RPC 2.0 communication over stdio streams.

Main components:
- ACPServer: Main server class for exposing agents via ACP JSON-RPC
- LLMlingACPAgent: Implementation of ACP Agent protocol for llmling agents
- ACPClientInterface: Client-side operations interface (filesystem, permissions, terminal)
- ACPSessionManager: Session lifecycle and state management
- Content converters: Handle format conversion between systems

ACP Features:
- JSON-RPC 2.0 communication over stdin/stdout streams using external acp library
- Session-oriented architecture with explicit lifecycle management
- Bidirectional agent-client communication
- Built-in filesystem operations and permission system
- Terminal integration for command execution
- Content blocks (text, image, audio, resource links)
- Standard compliance through external library's generated schemas
"""

from __future__ import annotations

from llmling_agent_acp.server import ACPServer, LLMlingACPAgent
from llmling_agent_acp.wrappers import ACPClientInterface, DefaultACPClient
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
from llmling_agent_acp.acp_types import StopReason


__all__ = [
    "ACPClientInterface",
    # Command bridge
    "ACPCommandBridge",
    "ACPServer",
    "ACPSession",
    "ACPSessionManager",
    "ACPToolBridge",
    "DefaultACPClient",
    "FileSystemBridge",
    "LLMlingACPAgent",
    # Stop reason type
    "StopReason",
    # Content types
    "convert_acp_mcp_server_to_config",
    "from_content_blocks",
    "to_content_blocks",
    "to_session_updates",
]
