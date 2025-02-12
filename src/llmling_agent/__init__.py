"""Agent configuration and creation."""

from llmling_agent.models import AgentsManifest, AgentConfig
from llmling_agent.agent import Agent, StructuredAgent, AnyAgent, AgentContext

from llmling_agent.delegation import AgentPool, Team, TeamRun
from dotenv import load_dotenv
from llmling_agent.messaging.messages import ChatMessage
from llmling_agent.tools import ToolInfo

__version__ = "0.99.5"

load_dotenv()

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentContext",
    "AgentPool",
    "AgentsManifest",
    "AnyAgent",
    "ChatMessage",
    "StructuredAgent",
    "Team",
    "TeamRun",
    "ToolInfo",
]
