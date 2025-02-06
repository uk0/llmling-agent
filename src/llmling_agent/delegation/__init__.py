"""Agent delegation and collaboration functionality."""

from llmling_agent.delegation.pool import AgentPool
from llmling_agent.delegation.base_team import BaseTeam
from llmling_agent.delegation.teamrun import TeamRun
from llmling_agent.delegation.injection import NodeInjectionError, inject_nodes
from llmling_agent.delegation.decorators import with_nodes
from llmling_agent.delegation.team import Team

__all__ = [
    "AgentPool",
    "BaseTeam",
    "NodeInjectionError",
    "Team",
    "TeamRun",
    "inject_nodes",
    "with_nodes",
]
