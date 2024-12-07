"""Agent configuration and creation."""

from llmling_agent.factory import create_agents_from_config
from llmling_agent.models import AgentDefinition, SystemPrompt
from llmling_agent.agent import LLMlingAgent


__version__ = "0.1.0"

__all__ = [
    "AgentDefinition",
    "LLMlingAgent",
    "SystemPrompt",
    "create_agents_from_config",
]
