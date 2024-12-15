"""Utilities for working with pydantic-ai types and objects."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypedDict

from pydantic_ai.messages import (
    Message,
    ModelStructuredResponse,
    ModelTextResponse,
    RetryPrompt,
    ToolReturn,
)

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


def format_response(response: (str | Message)) -> str:  # noqa: PLR0911
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
        case ModelTextResponse():
            return response.content
        case ModelStructuredResponse():
            try:
                calls = [
                    f"Tool: {call.tool_name}\nArgs: {call.args}"
                    for call in response.calls
                ]
                return "Tool Calls:\n" + "\n\n".join(calls)
            except Exception as e:  # noqa: BLE001
                msg = f"Could not format structured response: {e}"
                logger.warning(msg)
                return str(response)
        case ToolReturn():
            return f"Tool {response.tool_name} returned: {response.content}"
        case RetryPrompt():
            if isinstance(response.content, str):
                return f"Retry needed: {response.content}"
            return f"Validation errors:\n{json.dumps(response.content, indent=2)}"
        case _:
            return response.content


def find_last_assistant_message(messages: Sequence[Message]) -> str | None:
    """Find the last assistant message in history."""
    for msg in reversed(messages):
        match msg:
            case ModelTextResponse():
                return msg.content
            case ModelStructuredResponse():
                # Format structured response in a readable way
                calls = []
                for call in msg.calls:
                    if isinstance(call.args, dict):
                        args = call.args
                    else:
                        # Handle both ArgsJson and ArgsDict
                        args = (
                            call.args.args_dict  # type: ignore
                            if hasattr(call.args, "args_dict")
                            else call.args.args_json  # type: ignore
                        )
                    calls.append(f"Tool: {call.tool_name}\nArgs: {args}")
                return "\n\n".join(calls)
    return None
