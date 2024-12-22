"""Message and token usage models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field
from typing_extensions import TypeVar


T = TypeVar("T", str, BaseModel, default=str)


class TokenUsage(TypedDict):
    """Token usage statistics from model responses."""

    total: int
    """Total tokens used"""
    prompt: int
    """Tokens used in the prompt"""
    completion: int
    """Tokens used in the completion"""


@dataclass(frozen=True)
class TokenAndCostResult:
    """Combined token and cost tracking."""

    token_usage: TokenUsage
    """Token counts for prompt and completion"""
    cost_usd: float
    """Total cost in USD"""


class MessageMetadata(BaseModel):
    """Metadata for chat messages."""

    timestamp: datetime = Field(default_factory=datetime.now)
    model: str | None = Field(default=None)
    token_usage: TokenUsage | None = Field(default=None)
    cost: float | None = Field(default=None)
    tool: str | None = Field(default=None)
    # Add web UI specific fields
    avatar: str | None = Field(default=None)
    name: str | None = Field(default=None)
    tool_args: dict[str, Any] | None = None
    tool_result: Any | None = None
    model_config = {"frozen": True}


class ChatMessage[T](BaseModel):
    """Common message format for all UI types."""

    content: T
    model: str | None = Field(default=None)
    role: Literal["user", "assistant", "system"]
    metadata: MessageMetadata = Field(default_factory=MessageMetadata)
    timestamp: datetime = Field(default_factory=datetime.now)
    token_usage: TokenUsage | None = Field(default=None)
    message_id: str = Field(default_factory=lambda: str(uuid4()))

    model_config = {"frozen": True}

    def _get_content_str(self) -> str:
        """Get string representation of content."""
        match self.content:
            case str():
                return self.content
            case BaseModel():
                return self.content.model_dump_json(indent=2)
            case _:
                msg = f"Unexpected content type: {type(self.content)}"
                raise ValueError(msg)

    def to_gradio_format(self) -> tuple[str | None, str | None]:
        """Convert to Gradio chatbot format."""
        content_str = self._get_content_str()
        match self.role:
            case "user":
                return (content_str, None)
            case "assistant":
                return (None, content_str)
            case "system":
                return (None, f"System: {content_str}")
