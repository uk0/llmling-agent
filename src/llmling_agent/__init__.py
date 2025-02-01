"""Agent configuration and creation."""

from llmling_agent.models import AgentsManifest, AgentConfig, AgentContext
from llmling_agent.agent import Agent, StructuredAgent, AnyAgent
from llmling_agent.running import agent_function, run_agents_async

from llmling_agent.delegation import AgentPool, Team, TeamRun
from dotenv import load_dotenv
from llmling_agent.messaging.messages import ChatMessage

__version__ = "0.98.2"

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
    "agent_function",
    "run_agents_async",
]
