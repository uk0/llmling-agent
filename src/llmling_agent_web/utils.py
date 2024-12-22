"""UI state management for web interface."""

from __future__ import annotations

from pydantic_ai.messages import (
    ArgsDict,
    ModelMessage,
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

from llmling_agent_web.type_utils import GradioChatMessage


async def format_message_with_metadata(  # noqa: PLR0911
    message: ModelMessage | ModelRequestPart | ModelResponsePart,
) -> GradioChatMessage:
    """Format message with metadata for Gradio chat."""
    match message:
        case ModelResponse() as resp:
            # Handle each part in the response
            parts = []
            metadata = {}
            for part in resp.parts:
                match part:
                    case TextPart():
                        parts.append(part.content)
                    case ToolCallPart():
                        args = (
                            part.args.args_dict
                            if isinstance(part.args, ArgsDict)
                            else part.args.args_json
                        )
                        tool_info = f"🛠️ {part.tool_name}: {args}"
                        parts.append(tool_info)
                        metadata["tool"] = part.tool_name
            return GradioChatMessage(
                role="assistant",
                content="\n".join(parts) if parts else "",
                metadata=metadata,
            )
        case TextPart():
            return GradioChatMessage(
                role="assistant",
                content=message.content,
            )
        case ToolCallPart():
            args = (
                message.args.args_dict
                if isinstance(message.args, ArgsDict)
                else message.args.args_json
            )
            return GradioChatMessage(
                role="assistant",
                content=f"🛠️ Using tool: {message.tool_name}",
                metadata={"tool": message.tool_name, "args": args},
            )
        case ToolReturnPart():
            return GradioChatMessage(
                role="assistant",
                content=message.content,
                metadata={"title": f"🛠️ Used tool: {message.tool_name}"},
            )
        case RetryPromptPart():
            if isinstance(message.content, str):
                error_content = message.content
            else:
                error_content = "\n".join(
                    f"- {error['loc']}: {error['msg']}" for error in message.content
                )
            return GradioChatMessage(
                role="assistant",
                content=error_content,
                metadata={
                    "title": "⚠️ Retry needed",
                    "tool": message.tool_name if message.tool_name else None,
                },
            )
        case SystemPromptPart() | UserPromptPart():
            return GradioChatMessage(
                role="user" if isinstance(message, UserPromptPart) else "system",
                content=message.content,
            )
        case _:
            return GradioChatMessage(
                role="assistant",
                content=str(message),
            )
