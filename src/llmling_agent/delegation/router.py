"""Routing configuration and decision models."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from llmling_agent.agent import AnyAgent
    from llmling_agent.delegation.callbacks import DecisionCallback
    from llmling_agent.delegation.pool import AgentPool
    from llmling_agent.models.messages import ChatMessage


logger = logging.getLogger(__name__)


class Decision(BaseModel):
    """Base class for all routing decisions."""

    type: str = Field(init=False)
    """Discriminator field for decision types."""

    reason: str
    """Reason for this routing decision."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    async def execute(
        self,
        message: ChatMessage[Any],
        source_agent: AnyAgent[Any, Any],
        pool: AgentPool,
    ):
        """Execute this routing decision."""
        raise NotImplementedError


class RouteDecision(Decision):
    """Forward message without waiting for response."""

    type: Literal["route"] = Field("route", init=False)
    """Type discriminator for routing decisions."""

    target_agent: str
    """Name of the agent to forward the message to."""

    async def execute(
        self,
        message: ChatMessage[Any],
        source_agent: AnyAgent[Any, Any],
        pool: AgentPool,
    ):
        """Forward message and continue."""
        target = pool.get_agent(self.target_agent)
        target.outbox.emit(message, None)


class AwaitResponseDecision(Decision):
    """Forward message and wait for response."""

    type: Literal["await_response"] = Field("await_response", init=False)
    """Type discriminator for await decisions."""

    target_agent: str
    """Name of the agent to forward the message to."""

    talk_back: bool = False
    """Whether to send response back to original agent."""

    async def execute(
        self,
        message: ChatMessage[Any],
        source_agent: AnyAgent[Any, Any],
        pool: AgentPool,
    ):
        """Forward message and wait for response."""
        target = pool.get_agent(self.target_agent)
        response = await target.run(str(message))
        if self.talk_back:
            source_agent.outbox.emit(response, None)


class EndDecision(Decision):
    """End the conversation."""

    type: Literal["end"] = Field("end", init=False)
    """Type discriminator for end decisions."""

    async def execute(
        self,
        message: ChatMessage[Any],
        source_agent: AnyAgent[Any, Any],
        pool: AgentPool,
    ):
        """End the conversation."""


class AgentRouter:
    """Base class for routing messages between agents."""

    async def decide(self, message: Any) -> Decision:
        """Make routing decision for message."""
        raise NotImplementedError

    def get_wait_decision(
        self, target: str, reason: str, talk_back: bool = False
    ) -> Decision:
        """Create decision to route and wait for response."""
        return AwaitResponseDecision(
            target_agent=target, reason=reason, talk_back=talk_back
        )

    def get_route_decision(self, target: str, reason: str) -> Decision:
        """Create decision to route without waiting."""
        return RouteDecision(target_agent=target, reason=reason)

    def get_end_decision(self, reason: str) -> Decision:
        """Create decision to end routing."""
        return EndDecision(reason=reason)


class CallbackRouter[TMessage](AgentRouter):
    """Router using callback function for decisions."""

    def __init__(
        self,
        pool: AgentPool,
        decision_callback: DecisionCallback[TMessage],
    ):
        self.pool = pool
        self.decision_callback = decision_callback

    async def decide(self, message: TMessage) -> Decision:
        """Execute callback and handle sync/async appropriately."""
        result = self.decision_callback(message, self.pool, self)
        if inspect.isawaitable(result):
            return await result
        return result
