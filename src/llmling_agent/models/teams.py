"""Team configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from llmling_agent.models.forward_targets import ForwardingTarget  # noqa: TC001


class TeamConfig(BaseModel):
    """Configuration for a team or chain of message nodes.

    Teams can be either parallel execution groups or sequential chains.
    They can contain both agents and other teams as members.
    """

    mode: Literal["parallel", "sequential"]
    """Execution mode for team members."""

    members: list[str]
    """Names of agents or other teams that are part of this team."""

    connections: list[ForwardingTarget] | None = None
    """Optional message forwarding targets for this team."""

    shared_prompt: str | None = None
    """Optional shared prompt for this team."""

    # Future extensions:
    # tools: list[str] | None = None
    # """Tools available to all team members."""

    # knowledge: Knowledge | None = None
    # """Knowledge sources shared by all team members."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")
