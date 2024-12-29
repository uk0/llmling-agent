"""Agent configuration and creation."""

from llmling_agent.models import AgentsManifest, SystemPrompt
from llmling_agent.agent import LLMlingAgent
from llmling_agent.functional import (
    run_with_model,
    run_with_model_sync,
    run_agent_pipeline_sync,
    run_agent_pipeline,
    get_structured,
)
from llmling_agent.delegation.pool import AgentPool
from dotenv import load_dotenv

__version__ = "0.14.1"

load_dotenv()

__all__ = [
    "AgentPool",
    "AgentsManifest",
    "LLMlingAgent",
    "SystemPrompt",
    "get_structured",
    "run_agent_pipeline",
    "run_agent_pipeline_sync",
    "run_with_model",
    "run_with_model_sync",
]
