"""Team configuration models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from llmling_agent.events.sources import EventConfig  # noqa: TC001
from llmling_agent.models.forward_targets import ForwardingTarget  # noqa: TC001


class NodeConfig(BaseModel):
    """Configuration for a Node of the messaging system."""

    name: str | None = None
    """Name of the Agent / Team"""

    description: str | None = None
    """Optional description of the agent / team."""

    triggers: list[EventConfig] = Field(default_factory=list)
    """Event sources that activate this agent / team"""

    connections: list[ForwardingTarget] = Field(default_factory=list)
    """Targets to forward results to."""

    # Future extensions:
    # tools: list[str] | None = None
    # """Tools available to all team members."""

    # knowledge: Knowledge | None = None
    # """Knowledge sources shared by all team members."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        use_attribute_docstrings=True,
    )
