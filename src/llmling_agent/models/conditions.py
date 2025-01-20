from __future__ import annotations

from collections.abc import Awaitable, Callable  # noqa: TC003
from datetime import datetime, timedelta
import inspect
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ImportString


if TYPE_CHECKING:
    from llmling_agent.models.messages import ChatMessage
    from llmling_agent.talk.stats import TalkStats


class ConnectionCondition(BaseModel):
    """Base class for connection control conditions."""

    type: str = Field(init=False)
    """Discriminator for condition types."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    async def check(self, message: ChatMessage[Any], stats: TalkStats) -> bool:
        """Check if condition is met."""
        raise NotImplementedError


class WordMatchCondition(ConnectionCondition):
    """Disconnect when word/phrase is found in message."""

    type: Literal["word_match"] = Field("word_match", init=False)
    """Type discriminator."""

    words: list[str]
    """Words or phrases to match in messages."""

    case_sensitive: bool = False
    """Whether to match case-sensitively."""

    mode: Literal["any", "all"] = "any"
    """Match mode:
    - any: Trigger if any word matches
    - all: Require all words to match
    """

    async def check(self, message: ChatMessage[Any], stats: TalkStats) -> bool:
        """Check if message contains specified words."""
        text = str(message.content)
        if not self.case_sensitive:
            text = text.lower()
            words = [w.lower() for w in self.words]
        else:
            words = self.words

        matches = [w in text for w in words]
        return all(matches) if self.mode == "all" else any(matches)


class MessageCountCondition(ConnectionCondition):
    """Disconnect after N messages."""

    type: Literal["message_count"] = Field("message_count", init=False)
    """Type discriminator."""

    max_messages: int
    """Maximum number of messages before triggering."""

    count_mode: Literal["total", "per_agent"] = "total"
    """How to count messages:
    - total: All messages in conversation
    - per_agent: Messages from each agent separately
    """

    async def check(self, message: ChatMessage[Any], stats: TalkStats) -> bool:
        """Check if message count threshold is reached."""
        if self.count_mode == "total":
            return stats.message_count >= self.max_messages

        # Count per agent
        agent_messages = [m for m in stats.messages if m.name == message.name]
        return len(agent_messages) >= self.max_messages


class TimeCondition(ConnectionCondition):
    """Disconnect after time period."""

    type: Literal["time"] = Field("time", init=False)
    """Type discriminator."""

    duration: timedelta
    """How long the connection should stay active."""

    async def check(self, message: ChatMessage[Any], stats: TalkStats) -> bool:
        """Check if time duration has elapsed."""
        elapsed = datetime.now() - stats.start_time
        return elapsed >= self.duration


class TokenThresholdCondition(ConnectionCondition):
    """Disconnect after token threshold is reached."""

    type: Literal["token_threshold"] = Field("token_threshold", init=False)
    """Type discriminator."""

    max_tokens: int
    """Maximum number of tokens allowed."""

    count_type: Literal["total", "prompt", "completion"] = "total"
    """What tokens to count:
    - total: All tokens used
    - prompt: Only prompt tokens
    - completion: Only completion tokens
    """

    async def check(self, message: ChatMessage[Any], stats: TalkStats) -> bool:
        """Check if token threshold is reached."""
        if not message.cost_info:
            return False

        match self.count_type:
            case "total":
                return stats.token_count >= self.max_tokens
            case "prompt":
                return message.cost_info.token_usage["prompt"] >= self.max_tokens
            case "completion":
                return message.cost_info.token_usage["completion"] >= self.max_tokens


class CostCondition(ConnectionCondition):
    """Stop when cost threshold is reached."""

    type: Literal["cost"] = Field("cost", init=False)
    """Type discriminator."""

    max_cost: float
    """Maximum cost in USD."""

    async def check(self, message: ChatMessage[Any], stats: TalkStats) -> bool:
        """Check if cost limit is reached."""
        return stats.total_cost >= self.max_cost


class CostLimitCondition(ConnectionCondition):
    """Disconnect when cost limit is reached."""

    type: Literal["cost_limit"] = Field("cost_limit", init=False)
    """Type discriminator."""

    max_cost: float
    """Maximum cost in USD before triggering."""

    async def check(self, message: ChatMessage[Any], stats: TalkStats) -> bool:
        """Check if cost limit is reached."""
        if not message.cost_info:
            return False
        return float(message.cost_info.total_cost) >= self.max_cost


class CallableCondition(ConnectionCondition):
    """Custom predicate function."""

    type: Literal["callable"] = Field("callable", init=False)
    """Type discriminator."""

    predicate: ImportString[Callable[..., bool | Awaitable[bool]]]
    """Function to evaluate condition:
    Args:
        message: Current message being processed
        stats: Current connection statistics
    Returns:
        Whether condition is met
    """

    async def check(self, message: ChatMessage[Any], stats: TalkStats) -> bool:
        """Execute predicate function."""
        result = self.predicate(message, stats)
        if inspect.isawaitable(result):
            return await result
        return result


class AndCondition(ConnectionCondition):
    """Require all conditions to be met."""

    type: Literal["and"] = Field("and", init=False)
    """Type discriminator."""

    conditions: list[ConnectionCondition]
    """List of conditions to check."""

    async def check(self, message: ChatMessage[Any], stats: TalkStats) -> bool:
        """Check if all conditions are met."""
        results = [await c.check(message, stats) for c in self.conditions]
        return all(results)


class OrCondition(ConnectionCondition):
    """Require any condition to be met."""

    type: Literal["or"] = Field("or", init=False)
    """Type discriminator."""

    conditions: list[ConnectionCondition]
    """List of conditions to check."""

    async def check(self, message: ChatMessage[Any], stats: TalkStats) -> bool:
        """Check if any condition is met."""
        results = [await c.check(message, stats) for c in self.conditions]
        return any(results)


# Union type for condition validation
Condition = Annotated[
    WordMatchCondition
    | MessageCountCondition
    | TimeCondition
    | TokenThresholdCondition
    | CostLimitCondition
    | CallableCondition
    | AndCondition
    | OrCondition,
    Field(discriminator="type"),
]
