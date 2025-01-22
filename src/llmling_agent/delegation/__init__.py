"""Agent delegation and collaboration functionality."""

from llmling_agent.delegation.pool import AgentPool
from llmling_agent.delegation.execution import TeamRun, TeamRunMonitor, TeamRunStats
from llmling_agent.delegation.injection import AgentInjectionError, inject_agents
from llmling_agent.delegation.decorators import with_agents
from llmling_agent.delegation.agentgroup import Team, TeamResponse

__all__ = [
    "AgentInjectionError",
    "AgentPool",
    "Team",
    "TeamResponse",
    "TeamRun",
    "TeamRunMonitor",
    "TeamRunStats",
    "inject_agents",
    "with_agents",
]
