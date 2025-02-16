"""Team configuration models."""

from __future__ import annotations

from typing import Literal

from llmling_agent_config.nodes import NodeConfig


class TeamConfig(NodeConfig):
    """Configuration for a team or chain of message nodes.

    Teams can be either parallel execution groups or sequential chains.
    They can contain both agents and other teams as members.
    """

    mode: Literal["parallel", "sequential"]
    """Execution mode for team members."""

    members: list[str]
    """Names of agents or other teams that are part of this team."""

    shared_prompt: str | None = None
    """Optional shared prompt for this team."""

    # Future extensions:
    # tools: list[str] | None = None
    # """Tools available to all team members."""

    # knowledge: Knowledge | None = None
    # """Knowledge sources shared by all team members."""
