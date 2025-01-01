"""Agent configuration and creation."""

from llmling_agent.models import AgentsManifest, SystemPrompt, AgentConfig
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
from llmling_agent.models.messages import ChatMessage
from llmling_agent.chat_session.base import AgentPoolView

__version__ = "0.16.0"

load_dotenv()

__all__ = [
    "AgentConfig",
    "AgentPool",
    "AgentPoolView",
    "AgentsManifest",
    "ChatMessage",
    "LLMlingAgent",
    "SystemPrompt",
    "get_structured",
    "run_agent_pipeline",
    "run_agent_pipeline_sync",
    "run_with_model",
    "run_with_model_sync",
]
