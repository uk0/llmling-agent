"""Message and token usage models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict
from uuid import uuid4

from pydantic import BaseModel
from typing_extensions import TypeVar

from llmling_agent.common_types import JsonObject, MessageRole  # noqa: TC001
from llmling_agent.models.agents import ToolCallInfo  # noqa: TC001


TContent = TypeVar("TContent", str, BaseModel, default=str)


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
    total_cost: float
    """Total cost in USD"""


@dataclass
class ChatMessage[TContent]:
    """Common message format for all UI types.

    Generically typed with: ChatMessage[Type of Content]
    The type can either be str or a BaseModel subclass.
    """

    content: TContent
    """Message content, typed as TContent (either str or BaseModel)."""

    role: MessageRole
    """Role of the message sender (user/assistant/system)."""

    model: str | None = None
    """Name of the model that generated this message."""

    metadata: JsonObject = field(default_factory=dict)
    """Additional metadata about the message."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When this message was created."""

    cost_info: TokenAndCostResult | None = None
    """Token usage and costs for this specific message if available."""

    message_id: str = field(default_factory=lambda: str(uuid4()))
    """Unique identifier for this message."""

    response_time: float | None = None
    """Time it took the LLM to respond."""

    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    """List of tool calls made during message generation."""

    name: str | None = None
    """Display name for the message sender in UI."""

    forwarded_from: list[str] = field(default_factory=list)
    """List of agent names (the chain) that forwarded this message to the sender."""

    # model_config = ConfigDict(
    #     frozen=True, use_attribute_docstrings=True, arbitrary_types_allowed=True
    # )

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

    @property
    def data(self) -> TContent:
        """Get content as typed data. Provides compat to RunResult."""
        return self.content
