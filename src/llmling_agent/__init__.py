"""Agent configuration and creation."""

from llmling_agent.models import AgentsManifest, AgentConfig, AgentContext
from llmling_agent.agent import Agent, StructuredAgent

from llmling_agent.delegation import (
    AgentPool,
    Team,
    TeamRun,
    TeamRunStats,
    TeamRunMonitor,
)
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
    "AgentsManifest",
    "ChatMessage",
    "StructuredAgent",
    "Team",
    "TeamRun",
    "TeamRunMonitor",
    "TeamRunStats",
]
