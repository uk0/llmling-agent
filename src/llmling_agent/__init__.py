"""Agent configuration and creation."""

from llmling_agent.models import AgentsManifest, AgentConfig, AgentContext
from llmling_agent.agent import Agent, StructuredAgent

from llmling_agent.delegation import (
    AgentPool,
    Decision,
    EndDecision,
    RouteDecision,
    AwaitResponseDecision,
    interactive_controller,
    Team,
    TeamRun,
    TeamRunStats,
    TeamRunMonitor,
)
from llmling_agent.delegation.callbacks import DecisionCallback
from llmling_agent.delegation.router import AgentRouter, RuleRouter, CallbackRouter
from dotenv import load_dotenv
from llmling_agent.models.messages import ChatMessage
from llmling_agent.chat_session.base import AgentPoolView

__version__ = "0.94.0"

load_dotenv()

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentContext",
    "AgentPool",
    "AgentPoolView",
    "AgentRouter",
    "AgentsManifest",
    "AwaitResponseDecision",
    "CallbackRouter",
    "ChatMessage",
    "Decision",
    "DecisionCallback",
    "EndDecision",
    "RouteDecision",
    "RuleRouter",
    "StructuredAgent",
    "Team",
    "TeamRun",
    "TeamRunMonitor",
    "TeamRunStats",
    "interactive_controller",
]
