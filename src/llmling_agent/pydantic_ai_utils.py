"""Utilities for working with pydantic-ai types and objects."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypedDict

from pydantic_ai.messages import (
    ArgsDict,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.result import Cost

from llmling_agent.log import get_logger


logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.result import Cost


class TokenUsage(TypedDict):
    """Token usage statistics from model responses."""

    total: int
    """Total tokens used"""
    prompt: int
    """Tokens used in the prompt"""
    completion: int
    """Tokens used in the completion"""


def extract_token_usage(cost: Cost) -> TokenUsage | None:
    """Extract token usage statistics from a cost object.

    Args:
        cost: Cost object from model response

    Returns:
        Token usage statistics if available, None otherwise
    """
    if (
        cost
        and cost.total_tokens is not None
        and cost.request_tokens is not None
        and cost.response_tokens is not None
    ):
        return TokenUsage(
            total=cost.total_tokens,
            prompt=cost.request_tokens,
            completion=cost.response_tokens,
        )
    return None


def format_response(response: str | ModelMessage | ModelResponse) -> str:
    """Format any kind of response in a readable way.

    Args:
        response: Response to format.

    # TODO: Investigate if we should use result.new_messages() instead of
    # result.data for consistency with the streaming interface.

    Returns:
        A human-readable string representation
    """
    match response:
        case str():
            return response
        case ModelResponse() as resp:
            # Handle response parts
            parts = []
            for part in resp.parts:
                match part:
                    case TextPart():
                        parts.append(part.content)
                    case ToolCallPart() as tool_call:
                        args = (
                            tool_call.args.args_dict
                            if isinstance(tool_call.args, ArgsDict)
                            else json.loads(tool_call.args.args_json)
                        )
                        parts.append(f"Tool: {tool_call.tool_name}\nArgs: {args}")
            return "\n\n".join(parts) if parts else ""
        case _:
            # Fallback for other message types
            return str(response)


def find_last_assistant_message(messages: Sequence[ModelMessage]) -> str | None:
    """Find the last assistant message in history."""
    for msg in reversed(messages):
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                match part:
                    case TextPart():
                        return part.content
                    case ToolCallPart() as tool_call:
                        # Format tool calls nicely
                        args = (
                            tool_call.args.args_dict
                            if isinstance(tool_call.args, ArgsDict)
                            else json.loads(tool_call.args.args_json)
                        )
                        return f"Tool: {tool_call.tool_name}\nArgs: {args}"
    return None
