"""Utilities for working with pydantic-ai types and objects."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic_ai import messages as _messages, models
from pydantic_ai.messages import (
    ArgsDict,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
import tokonomics

from llmling_agent.log import get_logger
from llmling_agent.models.agents import ToolCallInfo
from llmling_agent.models.messages import ChatMessage, TokenAndCostResult, TokenUsage


logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.result import Usage

    from llmling_agent.tools.base import ToolInfo


async def extract_usage(
    usage: Usage,
    model: str,
    prompt: str,
    completion: str,
) -> TokenAndCostResult | None:
    """Extract token usage and calculate actual USD cost.

    Args:
        usage: Token counts from pydantic-ai Usage object
        model: Name of the model used
        prompt: The prompt text sent to model
        completion: The completion text received

    Returns:
        Token usage and USD cost, or None if counts unavailable
    """
    if not (
        usage
        and usage.total_tokens is not None
        and usage.request_tokens is not None
        and usage.response_tokens is not None
    ):
        logger.debug("Missing token counts in Usage object")
        return None

    token_usage = TokenUsage(
        total=usage.total_tokens,
        prompt=usage.request_tokens,
        completion=usage.response_tokens,
    )
    logger.debug("Token usage: %s", token_usage)

    cost = await tokonomics.calculate_token_cost(
        model,
        usage.request_tokens,
        usage.response_tokens,
    )
    total_cost = cost.total_cost if cost else 0.0
    return TokenAndCostResult(token_usage=token_usage, total_cost=total_cost)


def format_response(  # noqa: PLR0911
    response: str | _messages.ModelRequestPart | _messages.ModelResponsePart,
) -> str:
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


def get_tool_calls(
    messages: list[ModelMessage],
    tools: dict[str, ToolInfo] | None = None,
    context_data: Any | None = None,
) -> list[ToolCallInfo]:
    """Extract tool call information from messages.

    Args:
        messages: Messages from captured run
        tools: Original ToolInfo set to enrich ToolCallInfos with additional info
        context_data: Optional context data to attach to tool calls
    """
    tools = tools or {}
    parts = [part for message in messages for part in message.parts]
    call_parts = {
        part.tool_call_id: part
        for part in parts
        if isinstance(part, ToolCallPart) and part.tool_call_id
    }
    return [
        parts_to_tool_call_info(
            call_parts[part.tool_call_id], part, tools.get(part.tool_name), context_data
        )
        for part in parts
        if isinstance(part, ToolReturnPart) and part.tool_call_id in call_parts
    ]


def parts_to_tool_call_info(
    call_part: ToolCallPart,
    return_part: ToolReturnPart,
    tool_info: ToolInfo | None,
    context_data: Any | None = None,
) -> ToolCallInfo:
    """Convert matching tool call and return parts into a ToolCallInfo."""
    args = (
        call_part.args.args_dict
        if isinstance(call_part.args, ArgsDict)
        else json.loads(call_part.args.args_json)
    )

    return ToolCallInfo(
        tool_name=call_part.tool_name,
        args=args,
        result=return_part.content,
        tool_call_id=call_part.tool_call_id,
        timestamp=return_part.timestamp,
        context_data=context_data,
        agent_tool_name=tool_info.agent_name if tool_info else None,
    )


def convert_model_message(
    message: ModelMessage | ModelRequestPart | ModelResponsePart,
    tools: dict[str, ToolInfo] | None = None,
) -> ChatMessage:
    """Convert a pydantic-ai message to our ChatMessage format.

    Also supports converting parts of a message (with limitations then of course)

    Args:
        message: Message to convert (ModelMessage or its parts)
        tools: Original ToolInfo set to enrich ToolCallInfos with additional info

    Returns:
        Converted ChatMessage

    Raises:
        ValueError: If message type is not supported
    """
    match message:
        case ModelRequest():
            # Collect content from all parts
            content_parts = []
            role = "system"
            for part in message.parts:
                match part:
                    case UserPromptPart():
                        content_parts.append(str(part.content))
                        role = "user"
                    case SystemPromptPart():
                        content_parts.append(str(part.content))
            return ChatMessage(content="\n".join(content_parts), role=role)

        case ModelResponse():
            # Collect content and tool calls from all parts
            tool_calls = get_tool_calls([message], tools, None)
            parts = [format_response(p) for p in message.parts if isinstance(p, TextPart)]
            content = "\n".join(parts)
            return ChatMessage(content=content, role="assistant", tool_calls=tool_calls)

        case TextPart() | UserPromptPart() | SystemPromptPart() as part:
            role = "assistant" if isinstance(part, TextPart) else "user"
            return ChatMessage(content=format_response(part), role=role)

        case ToolCallPart():
            args = (
                message.args.args_dict
                if isinstance(message.args, ArgsDict)
                else json.loads(message.args.args_json)
            )
            info = ToolCallInfo(
                tool_name=message.tool_name,
                args=args,
                result=None,  # Not available yet
                tool_call_id=message.tool_call_id,
            )
            content = f"Tool call: {message.tool_name}\nArgs: {args}"
            return ChatMessage(content=content, role="assistant", tool_calls=[info])

        case ToolReturnPart():
            info = ToolCallInfo(
                tool_name=message.tool_name,
                args={},  # No args in return part
                result=message.content,
                tool_call_id=message.tool_call_id,
                timestamp=message.timestamp,
            )
            content = f"Tool {message.tool_name} returned: {message.content}"
            return ChatMessage(content=content, role="assistant", tool_calls=[info])

        case RetryPromptPart():
            error_content = (
                message.content
                if isinstance(message.content, str)
                else "\n".join(
                    f"- {error['loc']}: {error['msg']}" for error in message.content
                )
            )
            return ChatMessage(content=f"Retry needed: {error_content}", role="assistant")

        case _:
            msg = f"Unsupported message type: {type(message)}"
            raise ValueError(msg)


def models_equal(
    self, a: str | models.Model | None, b: str | models.Model | None
) -> bool:
    """Compare models by their string representation."""
    match (a, b):
        case (None, None):
            return True
        case (None, _) | (_, None):
            return False
        case _:
            # Compare string representations (using model.name() for Model objects)
            name_a = a if isinstance(a, str | None) else a.name()
            name_b = b if isinstance(b, str | None) else b.name()
            return name_a == name_b
