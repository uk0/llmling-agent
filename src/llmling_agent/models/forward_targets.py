from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class ForwardTarget(BaseModel):
    """Base model for message forwarding targets."""

    type: str = Field(init=False)


class AgentTarget(ForwardTarget):
    """Forward messages to another agent."""

    type: Literal["agent"] = Field("agent", init=False)
    name: str


ForwardingTarget = Annotated[
    AgentTarget,
    Field(discriminator="type"),
]
