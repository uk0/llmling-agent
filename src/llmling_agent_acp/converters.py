"""Content conversion utilities for ACP (Agent Client Protocol) integration.

This module handles conversion between llmling-agent message formats and ACP protocol
content blocks, session updates, and other data structures using the external acp library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from acp.acp_types import HttpMcpServer, SseMcpServer, StdioMcpServer
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    AudioContentBlock,
    ContentToolCallContent,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    PermissionOption,
    ResourceContentBlock,
    ResourceLink,
    SessionNotification,
    TextContentBlock,
    TextResourceContents,
    ToolCallLocation,
    ToolCallStart as ToolCall,
)
from llmling_agent.log import get_logger
from llmling_agent_config.mcp_server import (
    SSEMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from acp.acp_types import ContentBlock, MCPServer, ToolCallKind, ToolCallStatus
    from llmling_agent.models.content import BaseContent
    from llmling_agent_config.mcp_server import MCPServerConfig


logger = get_logger(__name__)


DEFAULT_PERMISSION_OPTIONS = [
    PermissionOption(option_id="allow_once", name="Allow Once", kind="allow_once"),
    PermissionOption(option_id="deny_once", name="Deny Once", kind="reject_once"),
    PermissionOption(option_id="allow_always", name="Always Allow", kind="allow_always"),
    PermissionOption(option_id="deny_always", name="Always Deny", kind="reject_always"),
]


@overload
def convert_acp_mcp_server_to_config(
    acp_server: HttpMcpServer,
) -> StreamableHTTPMCPServerConfig: ...


@overload
def convert_acp_mcp_server_to_config(
    acp_server: SseMcpServer,
) -> SSEMCPServerConfig: ...


@overload
def convert_acp_mcp_server_to_config(
    acp_server: StdioMcpServer,
) -> StdioMCPServerConfig: ...


@overload
def convert_acp_mcp_server_to_config(acp_server: MCPServer) -> MCPServerConfig: ...


def convert_acp_mcp_server_to_config(acp_server: MCPServer) -> MCPServerConfig:
    """Convert ACP McpServer to llmling MCPServerConfig.

    Args:
        acp_server: ACP McpServer object from session/new request

    Returns:
        MCPServerConfig instance
    """
    match acp_server:
        case StdioMcpServer():
            return StdioMCPServerConfig(
                name=acp_server.name,
                command=acp_server.command,
                args=list(acp_server.args),
                env={var.name: var.value for var in acp_server.env},
            )

        case SseMcpServer():
            return SSEMCPServerConfig(
                name=acp_server.name,
                url=acp_server.url,
                headers={h.name: h.value for h in acp_server.headers},
            )

        case HttpMcpServer():
            return StreamableHTTPMCPServerConfig(
                name=acp_server.name,
                url=acp_server.url,
                headers={h.name: h.value for h in acp_server.headers},
            )

        case _:
            msg = f"Unsupported MCP server type: {type(acp_server)}"
            raise ValueError(msg)


def format_uri_as_link(uri: str) -> str:
    """Format URI as markdown-style link similar to other ACP implementations.

    Args:
        uri: URI to format (file://, zed://, etc.)

    Returns:
        Markdown-style link in format [@name](uri)
    """
    if uri.startswith("file://"):
        path = uri[7:]  # Remove "file://"
        name = path.split("/")[-1] or path
        return f"[@{name}]({uri})"
    if uri.startswith("zed://"):
        parts = uri.split("/")
        name = parts[-1] or uri
        return f"[@{name}]({uri})"
    return uri


def from_content_blocks(blocks: Sequence[ContentBlock]) -> Sequence[str | BaseContent]:
    """Convert ACP content blocks to structured content objects.

    Args:
        blocks: List of ACP ContentBlock objects

    Returns:
        List of content objects (str for text, Content objects for rich media)
    """
    from llmling_agent.models.content import AudioBase64Content, ImageBase64Content

    content: list[str | BaseContent] = []

    for block in blocks:
        match block:
            case TextContentBlock():
                content.append(block.text)

            case ImageContentBlock():
                content.append(
                    ImageBase64Content(data=block.data, mime_type=block.mime_type)
                )

            case AudioContentBlock():
                # Audio always has data
                format_type = block.mime_type.split("/")[-1] if block.mime_type else "mp3"
                content.append(AudioBase64Content(data=block.data, format=format_type))

            case ResourceContentBlock():
                # Resource links - convert to text for now
                parts = [f"Resource: {block.name}"]
                if block.description:
                    parts.append(f"Description: {block.description}")
                parts.append(f"URI: {block.uri}")
                content.append("\n".join(parts))

            case ResourceLink():
                # Format as markdown-style link
                formatted_uri = format_uri_as_link(block.uri)
                content.append(formatted_uri)

            case EmbeddedResourceContentBlock():
                match block.resource:
                    case TextResourceContents():
                        uri = block.resource.uri
                        text = block.resource.text
                        formatted_uri = format_uri_as_link(uri)
                        content.append(formatted_uri)
                        context_block = f'\n<context ref="{uri}">\n{text}\n</context>'
                        content.append(context_block)
                    case _:
                        # Binary resource - just describe it with formatted URI
                        formatted_uri = format_uri_as_link(block.resource.uri)
                        content.append(f"Binary Resource: {formatted_uri}")

    return content


def to_agent_text_notification(
    response: str, session_id: str
) -> SessionNotification | None:
    """Convert agent response text to ACP session notification.

    Args:
        response: Response text from llmling agent
        session_id: ACP session identifier

    Returns:
        SessionNotification with agent text message, or None if response is empty
    """
    if not response.strip():
        return None

    update = AgentMessageChunk(content=TextContentBlock(text=response))
    return SessionNotification(session_id=session_id, update=update)


def infer_tool_kind(tool_name: str) -> ToolCallKind:  # noqa: PLR0911
    """Determine the appropriate tool kind based on name.

    Simple substring matching for tool kind inference.

    Args:
        tool_name: Name of the tool

    Returns:
        Tool kind string for ACP protocol
    """
    name_lower = tool_name.lower()
    if any(i in name_lower for i in ["read", "load", "get"]) and any(
        i in name_lower for i in ["file", "path", "content"]
    ):
        return "read"
    if any(
        i in name_lower for i in ["write", "save", "edit", "modify", "update"]
    ) and any(i in name_lower for i in ["file", "path", "content"]):
        return "edit"
    if any(i in name_lower for i in ["delete", "remove", "rm"]):
        return "delete"
    if any(i in name_lower for i in ["move", "rename", "mv"]):
        return "move"
    if any(i in name_lower for i in ["search", "find", "query", "lookup"]):
        return "search"
    if any(i in name_lower for i in ["execute", "run", "exec", "command", "shell"]):
        return "execute"
    if any(i in name_lower for i in ["think", "plan", "reason", "analyze"]):
        return "think"
    if any(i in name_lower for i in ["fetch", "download", "request"]):
        return "fetch"
    return "other"  # Default to other


def format_tool_call_for_acp(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: Any,
    session_id: str,
    status: ToolCallStatus = "completed",
    tool_call_id: str | None = None,
) -> SessionNotification:
    """Format tool execution as ACP tool call update.

    Args:
        tool_name: Name of the tool that was executed
        tool_input: Input parameters passed to the tool
        tool_output: Output returned by the tool
        session_id: ACP session identifier
        status: Execution status
        tool_call_id: Tool call identifier

    Returns:
        SessionNotification with tool call update
    """
    # Create tool call content from output
    content: list[ContentToolCallContent] = []
    if tool_output is not None:
        output_text = str(tool_output)
        block = TextContentBlock(text=output_text)
        content.append(ContentToolCallContent(content=block))

    # Extract file locations if present
    locations = [
        ToolCallLocation(path=value)
        for key, value in tool_input.items()
        if key in {"path", "file_path", "filepath"} and isinstance(value, str)
    ]

    tool_call = ToolCall(
        tool_call_id=tool_call_id or f"{tool_name}_{hash(str(tool_input))}",
        title=f"Execute {tool_name}",
        status=status,
        kind=infer_tool_kind(tool_name),
        locations=locations or None,
        content=content or None,
        raw_input=tool_input,
        raw_output=tool_output,
    )

    return SessionNotification(session_id=session_id, update=tool_call)


def create_thought_chunk(thought: str, session_id: str) -> SessionNotification:
    """Create an agent thought chunk for ACP streaming.

    Args:
        thought: Agent's internal thought/reasoning
        session_id: ACP session identifier

    Returns:
        SessionNotification with thought chunk
    """
    content = TextContentBlock(text=thought)
    update = AgentThoughtChunk(content=content)
    return SessionNotification(session_id=session_id, update=update)
