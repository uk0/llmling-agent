"""Utilities for working with pydantic-ai types and objects."""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, messages as _messages, models
from pydantic_ai.messages import (
    ArgsDict,
    ModelMessage,
    ModelRequest,
    ModelResponse,
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
    from collections.abc import Callable, Sequence

    from llmling.tools import LLMCallableTool
    from pydantic_ai.result import Usage


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
    return TokenAndCostResult(
        token_usage=token_usage, total_cost=cost.total_cost if cost else 0.0
    )


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
    messages: list[ModelMessage], context_data: Any | None = None
) -> list[ToolCallInfo]:
    """Extract tool call information from messages.

    Args:
        messages: Messages from captured run
        context_data: Optional context data to attach to tool calls
    """
    parts = [part for message in messages for part in message.parts]
    call_parts = {
        part.tool_call_id: part
        for part in parts
        if isinstance(part, ToolCallPart) and part.tool_call_id
    }
    return [
        parts_to_tool_call_info(call_parts[part.tool_call_id], part, context_data)
        for part in parts
        if isinstance(part, ToolReturnPart) and part.tool_call_id in call_parts
    ]


def parts_to_tool_call_info(
    call_part: ToolCallPart, return_part: ToolReturnPart, context_data: Any | None = None
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
    )


def convert_model_message(message: ModelMessage | Any) -> ChatMessage:  # noqa: PLR0911
    """Convert a pydantic-ai message to our ChatMessage format.

    Args:
        message: Message to convert (ModelMessage or its parts)

    Returns:
        Converted ChatMessage

    Raises:
        ValueError: If message type is not supported
    """
    match message:
        case ModelRequest():
            # Use first part's content
            part = message.parts[0]
            role = "user" if isinstance(part, UserPromptPart) else "system"
            return ChatMessage(content=str(part.content), role=role)

        case ModelResponse():
            # Convert first part (shouldn't have multiple typically)
            return convert_model_message(message.parts[0])

        case TextPart():
            return ChatMessage(content=message.content, role="assistant")

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

        case SystemPromptPart():
            return ChatMessage(content=message.content, role="system")

        case UserPromptPart():
            return ChatMessage(content=message.content, role="user")

        case _:
            msg = f"Unsupported message type: {type(message)}"
            raise ValueError(msg)


def register_tool(agent: Agent[Any, Any], tool: LLMCallableTool) -> None:
    """Register a tool with pydantic-ai agent using appropriate method.

    Args:
        agent: pydantic-ai agent to register with
        tool: Tool to register
    """

    def needs_context(func: Callable[..., Any]) -> bool:
        sig = inspect.signature(func)
        return any(
            "RunContext" in str(param.annotation) for param in sig.parameters.values()
        )

    assert tool._original_callable
    if needs_context(tool._original_callable):
        agent.tool(tool._original_callable)
    else:
        agent.tool_plain(tool._original_callable)


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
