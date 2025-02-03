"""Agent delegation and collaboration functionality."""

from llmling_agent.delegation.pool import AgentPool
from llmling_agent.delegation.base_team import BaseTeam
from llmling_agent.delegation.teamrun import TeamRun
from llmling_agent.delegation.injection import AgentInjectionError, inject_agents
from llmling_agent.delegation.decorators import with_agents
from llmling_agent.delegation.team import Team

__all__ = [
    "AgentInjectionError",
    "AgentPool",
    "BaseTeam",
    "Team",
    "TeamRun",
    "inject_agents",
    "with_agents",
]
