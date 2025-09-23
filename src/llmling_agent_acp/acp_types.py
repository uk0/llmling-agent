"""ACP (Agent Client Protocol) types re-exported from external library.

This module re-exports types from the external acp library to maintain
compatibility with existing code while using the standard ACP implementation.
"""

from __future__ import annotations

from typing import Literal

from acp.schema import (
    # Capability types
    ContentBlock1 as TextContent,
    ContentBlock2 as ImageContent,
    ContentBlock3 as AudioContent,
    ContentBlock4 as ResourceLink,
    ContentBlock5 as EmbeddedResource,
    # Terminal types
    RequestPermissionOutcome1,
    RequestPermissionOutcome2,
    # Permission types
    SessionUpdate1 as UserMessageChunk,
    SessionUpdate2 as AgentMessageChunk,
    SessionUpdate3 as AgentThoughtChunk,
    SessionUpdate4 as ToolCall,
    SessionUpdate5 as ToolCallUpdateMessage,
    SessionUpdate6 as Plan,
    SessionUpdate7 as AvailableCommandsUpdate,
    ToolCallContent1 as ToolCallContent,
    ToolCallContent2 as ToolCallDiffContent,
    ToolCallContent3 as ToolCallTerminalContent,
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
RequestPermissionOutcome = RequestPermissionOutcome1 | RequestPermissionOutcome2

# Tool call content union type
ToolCallContentUnion = ToolCallContent | ToolCallDiffContent | ToolCallTerminalContent


StopReason = Literal[
    "end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"
]
