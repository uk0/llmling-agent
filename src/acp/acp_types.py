"""ACP (Agent Client Protocol) types."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from acp.schema import (
    AgentMessageChunk,
    AgentPlan,
    AgentThoughtChunk,
    AllowedOutcome,
    AudioContentBlock,
    AvailableCommandsUpdate,
    ContentToolCallContent,
    DeniedOutcome,
    EmbeddedResourceContentBlock,
    FileEditToolCallContent,
    HttpMcpServer,
    ImageContentBlock,
    ResourceContentBlock,
    SseMcpServer,
    StdioMcpServer,
    TerminalToolCallContent,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
)


type ContentBlock = (
    TextContentBlock
    | ImageContentBlock
    | AudioContentBlock
    | ResourceContentBlock
    | EmbeddedResourceContentBlock
)

SessionUpdate = (
    UserMessageChunk
    | AgentMessageChunk
    | AgentThoughtChunk
    | ToolCallStart
    | ToolCallProgress
    | AgentPlan
    | AvailableCommandsUpdate
)
MCPServer = HttpMcpServer | SseMcpServer | StdioMcpServer
RequestPermissionOutcome = DeniedOutcome | AllowedOutcome
ToolCallContentUnion = (
    ContentToolCallContent | FileEditToolCallContent | TerminalToolCallContent
)


ToolCallStatus = Literal["pending", "in_progress", "completed", "failed"]
ConfirmationMode = Literal["confirm", "yolo", "human"]
StopReason = Literal[
    "end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"
]

# Plan entry types
PlanEntryPriority = Literal["high", "medium", "low"]
PlanEntryStatus = Literal["pending", "in_progress", "completed"]
ToolCallKind = Literal[
    "read",
    "edit",
    "delete",
    "move",
    "search",
    "execute",
    "think",
    "fetch",
    "switch_mode",
    "other",
]
JsonValue = Any
MethodHandler = Callable[[str, JsonValue | None, bool], Awaitable[JsonValue | None]]


@dataclass
class SessionData:
    """Complete session data for persistence and restoration."""

    session_id: str
    """Unique session identifier"""

    model_name: str
    """Currently chosen model"""

    session_mode: str | None
    """Currently selected session mode"""

    cwd: str
    """Working directory for the session"""

    mcp_servers: list[MCPServer]
    """MCP server configurations"""

    notifications: list[SessionUpdate]
    """Complete notification history"""

    metadata: dict[str, Any]
    """Additional session metadata"""
