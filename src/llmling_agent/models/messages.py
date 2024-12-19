"""Message and token usage models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypedDict

from pydantic import BaseModel, Field


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


# TODO: switch to this
# class TokenUsage(BaseModel):
#     """Token usage statistics from model responses."""

#     total: int
#     prompt: int
#     completion: int

#     model_config = {"frozen": True}


# @dataclass(frozen=True)
# class TokenAndCostResult:
#     """Combined token and cost tracking."""

#     token_usage: TokenUsage
#     cost_usd: float


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

    model_config = {"frozen": True}


class ChatMessage(BaseModel):
    """Common message format for all UI types."""

    content: str
    model: str | None = Field(default=None)
    role: Literal["user", "assistant", "system"]
    metadata: MessageMetadata | None = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.now)
    token_usage: TokenUsage | None = Field(default=None)

    model_config = {"frozen": True}

    def to_gradio_format(self) -> tuple[str | None, str | None]:
        """Convert to Gradio chatbot format."""
        match self.role:
            case "user":
                return (self.content, None)
            case "assistant":
                return (None, self.content)
            case "system":
                return (None, f"System: {self.content}")
