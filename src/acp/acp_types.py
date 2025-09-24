"""ACP (Agent Client Protocol) types re-exported from external library.

This module re-exports types from the external acp library to maintain
compatibility with existing code while using the standard ACP implementation.
"""

from __future__ import annotations

from typing import Literal

from acp.schema import (
    AgentMessageChunk,
    AgentPlan,
    AgentThoughtChunk,
    AllowedOutcome,
    AudioContentBlock,
    AvailableCommandsUpdate,
    ContentToolCallContent,
    # Terminal types
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
    # Permission types
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

# Session update union type
SessionUpdate = (
    UserMessageChunk
    | AgentMessageChunk
    | AgentThoughtChunk
    | ToolCallStart
    | ToolCallProgress
    | AgentPlan
    | AvailableCommandsUpdate
)

# Permission outcome union type
RequestPermissionOutcome = DeniedOutcome | AllowedOutcome

# Tool call content union type
ToolCallContentUnion = (
    ContentToolCallContent | FileEditToolCallContent | TerminalToolCallContent
)

MCPServer = HttpMcpServer | SseMcpServer | StdioMcpServer

StopReason = Literal[
    "end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"
]
