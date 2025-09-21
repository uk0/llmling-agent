"""ACP (Agent Client Protocol) types re-exported from external library.

This module re-exports types from the external acp library to maintain
compatibility with existing code while using the standard ACP implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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


if TYPE_CHECKING:
    from acp import RequestError


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


# Helper functions for JSON-RPC compatibility
def to_json_rpc_request(
    method: str, params: dict | None = None, request_id: int | None = None
) -> dict:
    """Create JSON-RPC 2.0 request."""
    request: dict[str, str | dict | int] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        request["params"] = params
    if request_id is not None:
        request["id"] = request_id
    return request


def to_json_rpc_response(
    result: dict | None = None,
    error: RequestError | None = None,
    request_id: int | None = None,
) -> dict:
    """Create JSON-RPC 2.0 response."""
    response: dict[str, str | dict | int | None] = {"jsonrpc": "2.0"}
    if request_id is not None:
        response["id"] = request_id
    if error:
        response["error"] = {
            "code": error.code,
            "message": str(error),
            "data": getattr(error, "data", None),
        }
    else:
        response["result"] = result
    return response


def to_json_rpc_notification(method: str, params: dict | None = None) -> dict:
    """Create JSON-RPC 2.0 notification."""
    notification: dict[str, str | dict] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        notification["params"] = params
    return notification


StopReason = Literal[
    "end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"
]
