"""Message and token usage models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict
from uuid import uuid4

from pydantic import BaseModel
import tokonomics
from typing_extensions import TypeVar

from llmling_agent.common_types import JsonObject, MessageRole  # noqa: TC001
from llmling_agent.log import get_logger
from llmling_agent.models.agents import ToolCallInfo  # noqa: TC001


TContent = TypeVar("TContent", str, BaseModel, default=str)

logger = get_logger(__name__)


class TokenUsage(TypedDict):
    """Token usage statistics from model responses."""

    total: int
    """Total tokens used"""
    prompt: int
    """Tokens used in the prompt"""
    completion: int
    """Tokens used in the completion"""


@dataclass(frozen=True)
class TokenCost:
    """Combined token and cost tracking."""

    token_usage: TokenUsage
    """Token counts for prompt and completion"""
    total_cost: float
    """Total cost in USD"""

    @classmethod
    async def from_usage(
        cls,
        usage: tokonomics.Usage | None,
        model: str,
        prompt: str,
        completion: str,
    ) -> TokenCost | None:
        """Create result from usage data.

        Args:
            usage: Token counts from model response
            model: Name of the model used
            prompt: The prompt text sent to model
            completion: The completion text received

        Returns:
            TokenCost if usage data available, None otherwise
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

        return cls(token_usage=token_usage, total_cost=total_cost)


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

    cost_info: TokenCost | None = None
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

    def to_text_message(self) -> ChatMessage[str]:
        """Convert this message to a text-only version."""
        return ChatMessage[str](
            content=str(self.content),
            role=self.role,
            name=self.name,
            model=self.model,
            metadata=self.metadata,
            timestamp=self.timestamp,
            cost_info=self.cost_info,
            message_id=self.message_id,
            response_time=self.response_time,
            tool_calls=self.tool_calls,
            forwarded_from=self.forwarded_from,
        )

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
