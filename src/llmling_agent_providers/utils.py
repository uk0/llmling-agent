from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from llmling_agent.messaging.messages import ChatMessage


def convert_message_to_chat(message: ChatMessage[Any]) -> list[dict[str, Any]]:
    """Convert message content to OpenAI chat format."""
    return [
        {
            "role": message.role,
            "content": str(message.content),
            "name": message.name if message.name else None,
        }
    ]
