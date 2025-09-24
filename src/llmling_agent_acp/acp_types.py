"""ACP (Agent Client Protocol) types re-exported from external library.

This module re-exports types from the external acp library to maintain
compatibility with existing code while using the standard ACP implementation.
"""

from __future__ import annotations

from typing import Literal

from acp.schema import (
    AgentMessageChunk,
    AgentPlan as Plan,
    AgentThoughtChunk,
    AllowedOutcome,
    AudioContentBlock as AudioContent,
    AvailableCommandsUpdate,
    ContentToolCallContent as ToolCallContent,
    # Terminal types
    DeniedOutcome,
    EmbeddedResourceContentBlock as EmbeddedResource,
    FileEditToolCallContent as ToolCallDiffContent,
    HttpMcpServer,
    ImageContentBlock as ImageContent,
    ResourceContentBlock as ResourceLink,
    SseMcpServer,
    StdioMcpServer,
    TerminalToolCallContent as ToolCallTerminalContent,
    TextContentBlock as TextContent,
    ToolCallProgress as ToolCallUpdateMessage,
    # Permission types
    ToolCallStart as ToolCall,
    UserMessageChunk,
)


# Content block union type
ContentBlock = TextContent | ImageContent | AudioContent | ResourceLink | EmbeddedResource

# Session update union type
SessionUpdate = (
    UserMessageChunk
    | AgentMessageChunk
    | AgentThoughtChunk
    | ToolCall
    | ToolCallUpdateMessage
    | Plan
    | AvailableCommandsUpdate
)

# Permission outcome union type
RequestPermissionOutcome = DeniedOutcome | AllowedOutcome

# Tool call content union type
ToolCallContentUnion = ToolCallContent | ToolCallDiffContent | ToolCallTerminalContent

MCPServer = HttpMcpServer | SseMcpServer | StdioMcpServer

StopReason = Literal[
    "end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"
]
