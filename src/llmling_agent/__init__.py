"""Agent configuration and creation."""

from llmling_agent.factory import create_agents_from_config
from llmling_agent.loader import load_agent_config_file
from llmling_agent.models import AgentDefinition, SystemPrompt
from llmling_agent.agent import LLMlingAgent


__version__ = "0.0.1"

__all__ = [
    "AgentDefinition",
    "LLMlingAgent",
    "SystemPrompt",
    "create_agents_from_config",
    "load_agent_config_file",
]
