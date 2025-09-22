"""Content conversion utilities for ACP (Agent Client Protocol) integration.

This module handles conversion between llmling-agent message formats and ACP protocol
content blocks, session updates, and other data structures using the external acp library.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from acp import (
    ReadTextFileRequest,
    RequestPermissionRequest,
    WriteTextFileRequest,
)
from acp.schema import (
    ContentBlock1,
    ContentBlock2,
    ContentBlock3,
    ContentBlock4,
    ContentBlock5,
    McpServer,
    PermissionOption,
    SessionNotification,
    SessionUpdate2 as AgentMessageChunk,
    SessionUpdate3 as AgentThoughtChunk,
    SessionUpdate4 as ToolCall,
    TextResourceContents,
    ToolCallContent1 as ToolCallContent,
    ToolCallLocation,
    ToolCallUpdate,
)

from llmling_agent.log import get_logger
from llmling_agent_config.mcp_server import StdioMCPServerConfig


# Define ContentBlock union type
type ContentBlock = (
    ContentBlock1 | ContentBlock2 | ContentBlock3 | ContentBlock4 | ContentBlock5
)


if TYPE_CHECKING:
    from llmling_agent.messaging.messages import ChatMessage

logger = get_logger(__name__)


def convert_acp_mcp_server_to_config(acp_server: McpServer) -> StdioMCPServerConfig:
    """Convert ACP McpServer to llmling MCPServerConfig.

    Args:
        acp_server: ACP McpServer object from session/new request

    Returns:
        StdioMCPServerConfig instance
    """
    # Convert environment variables
    env_dict = {var.name: var.value for var in acp_server.env}

    return StdioMCPServerConfig(
        name=acp_server.name,
        command=acp_server.command,
        args=acp_server.args,
        environment=env_dict,
    )


def from_content_blocks(blocks: list[ContentBlock]) -> str:
    """Convert ACP content blocks to a single prompt string for llmling agents.

    Args:
        blocks: List of ACP ContentBlock objects

    Returns:
        Combined prompt string suitable for llmling agents
    """
    parts = []

    for block in blocks:
        match block:
            case ContentBlock1():  # Text content
                parts.append(block.text)
            case ContentBlock2():  # Image content
                parts.append(f"[Image: {block.mimeType}]")
                if block.uri:
                    parts.append(f"Image URI: {block.uri}")
            case ContentBlock3():  # Audio content
                parts.append(f"[Audio: {block.mimeType}]")
            case ContentBlock4():  # Resource link
                parts.append(f"[Resource: {block.name}]")
                if block.description:
                    parts.append(f"Description: {block.description}")
                parts.append(f"URI: {block.uri}")
            case ContentBlock5():  # Embedded resource
                parts.append(f"[Resource: {block.resource.uri}]")
                match block.resource:
                    case TextResourceContents():
                        parts.append(block.resource.text)
                    case _:
                        parts.append("[Binary Resource]")

    return "\n".join(parts) if parts else ""


def to_content_blocks(text: str) -> list[ContentBlock]:
    """Convert text response to ACP content blocks.

    Args:
        text: Text response from llmling agent

    Returns:
        List of ACP ContentBlock objects
    """
    if not text.strip():
        return []

    # For now, return simple text content block
    # Future enhancement: detect and parse structured content
    return [ContentBlock1(text=text, type="text")]


def to_session_updates(response: str, session_id: str) -> list[SessionNotification]:
    """Convert agent response to ACP session update notifications.

    Args:
        response: Response text from llmling agent
        session_id: ACP session identifier

    Returns:
        List of SessionNotification objects for streaming to client
    """
    if not response.strip():
        return []

    # Split response into chunks for streaming
    chunks = _split_response_into_chunks(response)
    updates = []

    for chunk in chunks:
        if chunk.strip():
            content = ContentBlock1(text=chunk, type="text")
            update = AgentMessageChunk(
                content=content, sessionUpdate="agent_message_chunk"
            )
            updates.append(SessionNotification(sessionId=session_id, update=update))

    return updates


def to_chat_message(
    blocks: list[ContentBlock],
    agent_name: str = "user",
) -> ChatMessage[str]:
    """Convert ACP content blocks to llmling ChatMessage.

    Args:
        blocks: List of ACP ContentBlock objects
        agent_name: Name for the message sender

    Returns:
        llmling ChatMessage
    """
    from llmling_agent.messaging.messages import ChatMessage

    content = from_content_blocks(blocks)
    return ChatMessage[str](content=content, role="user", name=agent_name)


def from_chat_message(message: ChatMessage[Any]) -> list[ContentBlock]:
    """Convert llmling ChatMessage to ACP content blocks.

    Args:
        message: llmling ChatMessage

    Returns:
        List of ACP ContentBlock objects
    """
    content = str(message.content) if message.content is not None else ""
    return to_content_blocks(content)


def extract_file_references(text: str) -> list[dict[str, Any]]:
    """Extract file references from agent response text.

    Args:
        text: Response text to analyze

    Returns:
        List of file reference dictionaries
    """
    # Look for file path patterns
    file_patterns = [
        r"(?:file|path|wrote to|created|reading from):\s*([^\s\n]+)",
        r"`([^`]+\.[a-zA-Z0-9]+)`",  # Files in backticks with extensions
        r'"([^"]+\.[a-zA-Z0-9]+)"',  # Files in quotes with extensions
    ]

    files: list[dict[str, Any]] = []
    for pattern in file_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        files.extend(
            {"path": match, "type": "file_reference"}
            for match in matches
            if match and not match.startswith("http")  # Exclude URLs
        )

    return files


def _split_response_into_chunks(response: str, chunk_size: int = 100) -> list[str]:
    """Split response text into chunks for streaming.

    Args:
        response: Text to split
        chunk_size: Target size for each chunk

    Returns:
        List of text chunks
    """
    if len(response) <= chunk_size:
        return [response]

    chunks = []
    words = response.split()
    current_chunk: list[str] = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for space

        if current_length + word_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


class FileSystemBridge:
    """Bridges agent file operations with ACP client filesystem.

    This is a simplified bridge that provides utility functions for file operations.
    The actual ACP client communication should be handled at the session level.
    """

    def __init__(self) -> None:
        """Initialize filesystem bridge."""

    @staticmethod
    def create_read_request(
        path: str,
        session_id: str,
        line: int | None = None,
        limit: int | None = None,
    ) -> ReadTextFileRequest:
        """Create a read file request.

        Args:
            path: File path to read
            session_id: ACP session identifier
            line: Optional starting line number
            limit: Optional limit on number of lines

        Returns:
            ReadTextFileRequest object
        """
        return ReadTextFileRequest(
            sessionId=session_id,
            path=path,
            line=line,
            limit=limit,
        )

    @staticmethod
    def create_write_request(
        path: str, content: str, session_id: str
    ) -> WriteTextFileRequest:
        """Create a write file request.

        Args:
            path: File path to write
            content: Content to write
            session_id: ACP session identifier

        Returns:
            WriteTextFileRequest object
        """
        return WriteTextFileRequest(sessionId=session_id, path=path, content=content)

    @staticmethod
    def create_permission_request(
        operation: str, details: dict[str, Any], session_id: str
    ) -> RequestPermissionRequest:
        """Create a permission request for filesystem operation.

        Args:
            operation: Type of operation (read, write, etc.)
            details: Operation details
            session_id: ACP session identifier

        Returns:
            RequestPermissionRequest object
        """
        # Create permission request
        tool_call = ToolCallUpdate(
            toolCallId=f"fs_{operation}_{hash(details.get('path', ''))}",
            title=f"File {operation.title()}",
            status="pending_permission",
            kind="filesystem",
        )

        options = [
            PermissionOption(optionId="allow", name="Allow", kind="permission"),
            PermissionOption(optionId="deny", name="Deny", kind="permission"),
        ]

        return RequestPermissionRequest(
            sessionId=session_id,
            toolCall=tool_call,
            options=options,
        )


def _determine_tool_kind(tool_name: str) -> str:  # noqa: PLR0911
    """Determine the appropriate tool kind based on name.

    Args:
        tool_name: Name of the tool

    Returns:
        Tool kind string for ACP protocol
    """
    name_lower = tool_name.lower()

    # File operations
    if any(keyword in name_lower for keyword in ["read", "load", "get"]) and any(
        keyword in name_lower for keyword in ["file", "path", "content"]
    ):
        return "read"

    if any(
        keyword in name_lower for keyword in ["write", "save", "edit", "modify", "update"]
    ) and any(keyword in name_lower for keyword in ["file", "path", "content"]):
        return "edit"

    if any(keyword in name_lower for keyword in ["delete", "remove", "rm"]):
        return "delete"

    if any(keyword in name_lower for keyword in ["move", "rename", "mv"]):
        return "move"

    # Operations
    if any(keyword in name_lower for keyword in ["search", "find", "query", "lookup"]):
        return "search"

    if any(
        keyword in name_lower
        for keyword in ["execute", "run", "exec", "command", "shell"]
    ):
        return "execute"

    if any(keyword in name_lower for keyword in ["think", "plan", "reason", "analyze"]):
        return "think"

    if any(keyword in name_lower for keyword in ["fetch", "download", "request"]):
        return "fetch"

    # Default to other
    return "other"


def format_tool_call_for_acp(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_output: Any,
    session_id: str,
    status: str = "completed",
) -> SessionNotification:
    """Format tool execution as ACP tool call update.

    Args:
        tool_name: Name of the tool that was executed
        tool_input: Input parameters passed to the tool
        tool_output: Output returned by the tool
        session_id: ACP session identifier
        status: Execution status

    Returns:
        SessionNotification with tool call update
    """
    # Create tool call content from output
    content = []
    if tool_output is not None:
        output_text = str(tool_output)
        block = ContentBlock1(text=output_text, type="text")
        content.append(ToolCallContent(type="content", content=block))

    # Extract file locations if present
    locations = []
    for key, value in tool_input.items():
        if key in ("path", "file_path", "filepath") and isinstance(value, str):
            locations.append(ToolCallLocation(path=value))

    # Use simple execute title but infer kind for UI hints
    title = f"Execute {tool_name}"
    kind = _determine_tool_kind(tool_name)

    tool_call = ToolCall(
        toolCallId=f"{tool_name}_{hash(str(tool_input))}",
        title=title,
        status=status,
        kind=kind,
        locations=locations or None,
        content=content or None,
        rawInput=tool_input,
        rawOutput=tool_output,
        sessionUpdate="tool_call",
    )

    return SessionNotification(sessionId=session_id, update=tool_call)


def create_thought_chunk(thought: str, session_id: str) -> SessionNotification:
    """Create an agent thought chunk for ACP streaming.

    Args:
        thought: Agent's internal thought/reasoning
        session_id: ACP session identifier

    Returns:
        SessionNotification with thought chunk
    """
    content = ContentBlock1(text=thought, type="text")
    update = AgentThoughtChunk(content=content, sessionUpdate="agent_thought_chunk")
    return SessionNotification(sessionId=session_id, update=update)
