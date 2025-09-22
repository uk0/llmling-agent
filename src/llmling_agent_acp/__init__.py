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

# Re-export important types from external library
from acp import Agent as ACPAgent
from acp.schema import (
    # Core protocol types
    InitializeRequest,
    InitializeResponse,
    NewSessionRequest,
    NewSessionResponse,
    PromptRequest,
    PromptResponse,
    SessionNotification,
    # Content types
    ContentBlock1 as TextContent,
    ContentBlock2 as ImageContent,
    ContentBlock3 as AudioContent,
    ContentBlock4 as ResourceLink,
    ContentBlock5 as EmbeddedResource,
    # Capability types
    AgentCapabilities,
    ClientCapabilities,
    PromptCapabilities,
)

# Re-export protocol constants
from acp import (
    PROTOCOL_VERSION,
    AGENT_METHODS,
    CLIENT_METHODS,
)

__all__ = [
    "AGENT_METHODS",
    "CLIENT_METHODS",
    # Protocol constants
    "PROTOCOL_VERSION",
    # Protocol interfaces
    "ACPAgent",
    "ACPClientInterface",
    # Command bridge
    "ACPCommandBridge",
    "ACPServer",
    "ACPSession",
    "ACPSessionManager",
    "ACPToolBridge",
    "AgentCapabilities",
    "AudioContent",
    "ClientCapabilities",
    "DefaultACPClient",
    "EmbeddedResource",
    "FileSystemBridge",
    "ImageContent",
    "InitializeRequest",
    "InitializeResponse",
    "LLMlingACPAgent",
    "NewSessionRequest",
    "NewSessionResponse",
    "PromptCapabilities",
    "PromptRequest",
    "PromptResponse",
    "ResourceLink",
    "SessionNotification",
    # Stop reason type
    "StopReason",
    # Content types
    "TextContent",
    "convert_acp_mcp_server_to_config",
    "from_content_blocks",
    "to_content_blocks",
    "to_session_updates",
]
