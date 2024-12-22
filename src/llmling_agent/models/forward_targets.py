from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ForwardTarget(BaseModel):
    """Base model for message forwarding targets."""

    type: str = Field(init=False)
    """Discriminator field for forward target types."""

    wait_for_completion: bool = Field(True)
    """Whether to wait for the result before continuing.

    If True, message processing will wait for the target to complete.
    If False, message will be forwarded asynchronously.
    """
    model_config = ConfigDict(use_attribute_docstrings=True, frozen=True)


class AgentTarget(ForwardTarget):
    """Forward messages to another agent."""

    type: Literal["agent"] = Field("agent", init=False)
    """Type discriminator for agent targets."""

    name: str
    """Name of the agent to forward messages to."""


ForwardingTarget = Annotated[AgentTarget, Field(discriminator="type")]
