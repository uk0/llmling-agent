"""Message and token usage models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime  # noqa: TC003
from typing import Literal

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage statistics from model responses."""

    total: int
    prompt: int
    completion: int

    model_config = {"frozen": True}


@dataclass(frozen=True)
class TokenAndCostResult:
    """Combined token and cost tracking."""

    token_usage: TokenUsage
    cost_usd: float


class MessageMetadata(BaseModel):
    """Metadata for chat messages."""

    timestamp: datetime | None = Field(default=None)
    model: str | None = Field(default=None)
    token_usage: TokenUsage | None = Field(default=None)
    cost: float | None = Field(default=None)
    tool: str | None = Field(default=None)

    model_config = {"frozen": True}


class ChatMessage(BaseModel):
    """Common message format."""

    content: str
    role: Literal["user", "assistant", "system"]
    metadata: MessageMetadata | None = Field(default=None)

    model_config = {"frozen": True}
