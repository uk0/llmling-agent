"""Agent delegation and collaboration functionality."""

from llmling_agent.delegation.pool import AgentPool
from llmling_agent.delegation.router import AgentRouter, CallbackRouter, RuleRouter
from llmling_agent.delegation.controllers import interactive_controller
from llmling_agent.delegation.router import (
    Decision,
    EndDecision,
    RouteDecision,
    AwaitResponseDecision,
    RoutingRule,
    RoutingConfig,
)
from llmling_agent.delegation.callbacks import DecisionCallback
from llmling_agent.delegation.execution import TeamRun, TeamRunMonitor, TeamRunStats
from llmling_agent.delegation.injection import AgentInjectionError, inject_agents
from llmling_agent.delegation.decorators import with_agents
from llmling_agent.delegation.agentgroup import Team, TeamResponse

__all__ = [
    "AgentInjectionError",
    "AgentPool",
    "AgentRouter",
    "AwaitResponseDecision",
    "CallbackRouter",
    "Decision",
    "DecisionCallback",
    "EndDecision",
    "RouteDecision",
    "RoutingConfig",
    "RoutingRule",
    "RuleRouter",
    "Team",
    "TeamResponse",
    "TeamRun",
    "TeamRunMonitor",
    "TeamRunStats",
    "inject_agents",
    "interactive_controller",
    "with_agents",
]
