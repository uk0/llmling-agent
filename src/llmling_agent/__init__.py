"""Agent configuration and creation."""

from llmling_agent.models import AgentsManifest, AgentConfig, AgentContext
from llmling_agent.agent import Agent, StructuredAgent, AnyAgent

from llmling_agent.delegation import AgentPool, Team, TeamRun
from dotenv import load_dotenv
from llmling_agent.models.messages import ChatMessage
from llmling_agent.chat_session.base import AgentPoolView

__version__ = "0.96.0"

load_dotenv()

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentContext",
    "AgentPool",
    "AgentPoolView",
    "AgentsManifest",
    "AnyAgent",
    "ChatMessage",
    "StructuredAgent",
    "Team",
    "TeamRun",
]
