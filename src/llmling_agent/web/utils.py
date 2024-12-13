"""UI state management for web interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    Message,
    ModelStructuredResponse,
    ModelTextResponse,
    RetryPrompt,
    ToolReturn,
    UserPrompt,
)


if TYPE_CHECKING:
    from llmling_agent.web.type_utils import ChatMessage


async def format_message_with_metadata(message: Message) -> ChatMessage:
    """Format message with metadata for Gradio chat."""
    match message:
        case ModelStructuredResponse():
            # Show tool calls
            tool_calls = []
            for call in message.calls:
                args = (
                    call.args.args_dict
                    if hasattr(call.args, "args_dict")
                    else call.args.args_json
                )
                tool_calls.append(f"🛠️ {call.tool_name}: {args}")
            return {
                "role": "assistant",
                "content": "",  # Will be filled by actual response
                "metadata": {"title": "Thinking...", "tools": "\n".join(tool_calls)},
            }
        case ToolReturn():
            return {
                "role": "assistant",
                "content": message.content,
                "metadata": {"title": f"🛠️ Used tool: {message.tool_name}"},
            }
        case ModelTextResponse():
            return {
                "role": "assistant",
                "content": message.content,
            }
        case RetryPrompt():
            # Format validation errors or retry messages nicely
            if isinstance(message.content, str):
                error_content = message.content
            else:
                # Format validation errors
                errors = [
                    f"- {error['loc']}: {error['msg']}" for error in message.content
                ]
                error_content = "\n".join(errors)

            return {
                "role": "assistant",
                "content": error_content,
                "metadata": {
                    "title": "⚠️ Retry needed",
                    "tool": message.tool_name if message.tool_name else None,
                },
            }
        # case SystemPrompt():
        #     return {
        #         "role": "system",
        #         "content": message.content,
        #         "metadata": {"title": "System Message"},
        #     }
        case UserPrompt():
            return {
                "role": "user",
                "content": message.content,
            }
        case _:
            # Fallback for any other message types
            return {
                "role": message.role,
                "content": str(message),
            }
