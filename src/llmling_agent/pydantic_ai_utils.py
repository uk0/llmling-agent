"""Utilities for working with pydantic-ai types and objects."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic_ai import messages as _messages
from pydantic_ai.messages import (
    ArgsDict,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)

from llmling_agent.costs.core import calculate_token_cost
from llmling_agent.log import get_logger
from llmling_agent.models.messages import TokenAndCostResult, TokenUsage


logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.result import Usage


async def extract_token_usage_and_cost(
    cost: Usage,
    model: str,
    prompt: str,
    completion: str,
) -> TokenAndCostResult | None:
    """Extract token usage and calculate actual USD cost.

    Args:
        cost: Token counts from pydantic-ai Cost object
        model: Name of the model used
        prompt: The prompt text sent to model
        completion: The completion text received

    Returns:
        Token usage and USD cost, or None if counts unavailable
    """
    if not (
        cost
        and cost.total_tokens is not None
        and cost.request_tokens is not None
        and cost.response_tokens is not None
    ):
        logger.debug("Missing token counts in cost object")
        return None

    token_usage = TokenUsage(
        total=cost.total_tokens,
        prompt=cost.request_tokens,
        completion=cost.response_tokens,
    )
    logger.debug("Token usage: %s", token_usage)

    cost_usd = await calculate_token_cost(model, token_usage)
    if cost_usd is not None:
        logger.debug("Calculated cost: $%.6f", cost_usd)
        return TokenAndCostResult(token_usage=token_usage, cost_usd=cost_usd)

    logger.debug("Failed to calculate USD cost")
    return None


def format_response(response: str | _messages.ModelRequestPart) -> str:  # noqa: PLR0911
    """Format any kind of response in a readable way.

    Args:
        response: Response part to format

    Returns:
        A human-readable string representation
    """
    match response:
        case str():
            return response
        case _messages.TextPart():
            return response.content
        case _messages.ToolCallPart():
            if isinstance(response.args, _messages.ArgsJson):
                args = response.args.args_json
            else:
                args = str(response.args.args_dict)
            return f"Tool call: {response.tool_name}\nArgs: {args}"
        case _messages.ToolReturnPart():
            return f"Tool {response.tool_name} returned: {response.content}"
        case _messages.RetryPromptPart():
            if isinstance(response.content, str):
                return f"Retry needed: {response.content}"
            return f"Validation errors:\n{response.content}"
        case _:
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
