"""Routing configuration and decision models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Decision(BaseModel):
    """Base class for all routing decisions."""

    type: str
    """Discriminator field for decision types."""

    reason: str
    """Reason for this routing decision."""

    model_config = ConfigDict(use_attribute_docstrings=True)


class RouteDecision(Decision):
    """Forward message to another agent without waiting.

    The message will be sent to the target agent, but execution continues
    immediately without waiting for a response.
    """

    type: Literal["route"] = "route"
    """Type discriminator for routing decisions."""

    target_agent: str
    """Name of the agent to forward the message to."""


class AwaitResponseDecision(Decision):
    """Forward message to another agent and await response.

    The message will be sent to the target agent and execution will pause
    until the target agent responds. Used when the response is needed
    for further processing.
    """

    type: Literal["await_response"] = "await_response"
    """Type discriminator for await decisions."""

    target_agent: str
    """Name of the agent to forward the message to and await response from."""


class EndDecision(Decision):
    """End the conversation.

    Signal that no further routing is needed and the conversation
    can be considered complete.
    """

    type: Literal["end"] = "end"
    """Type discriminator for end decisions."""


# Routing configuration
@dataclass
class RoutingRule:
    """Rule for routing messages based on content matching.

    Defines when and how messages should be routed to specific agents.
    Rules are evaluated in priority order, with the first matching rule
    being applied.
    """

    keyword: str
    """Keyword to match in the message content."""

    target: str
    """Name of the agent to route to when rule matches."""

    reason: str
    """Reason for this routing decision."""

    wait_for_response: bool = True
    """Whether to wait for the target agent's response."""

    priority: int = 100
    """Rule priority (lower numbers = higher priority)."""

    requires_capability: str | None = None
    """Optional capability the target agent must have."""


class RoutingConfig(BaseModel):
    """Complete routing configuration for an agent.

    Defines a set of rules for message routing and default behavior
    when no rules match. Rules are evaluated in priority order.
    """

    rules: list[RoutingRule]
    """Ordered list of routing rules to evaluate."""

    default_target: str | None = None
    """Agent to route to when no rules match."""

    default_reason: str = "No specific rule matched"
    """Reason to use for default routing."""

    case_sensitive: bool = False
    """Whether keyword matching should be case-sensitive."""

    model_config = ConfigDict(use_attribute_docstrings=True)
