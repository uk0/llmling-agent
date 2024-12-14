"""Agent configuration and creation."""

from llmling_agent.models import AgentsManifest, SystemPrompt
from llmling_agent.agent import LLMlingAgent
from llmling_agent.runners import AgentOrchestrator, AgentRunConfig, SingleAgentRunner
from dotenv import load_dotenv

__version__ = "0.5.1"

load_dotenv()

__all__ = [
    "AgentOrchestrator",
    "AgentRunConfig",
    "AgentsManifest",
    "LLMlingAgent",
    "SingleAgentRunner",
    "SystemPrompt",
]
