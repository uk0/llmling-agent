"""Agent configuration and creation."""

from llmling_agent.models import AgentsManifest, SystemPrompt, AgentConfig
from llmling_agent.agent import Agent, HumanAgent
from llmling_agent.functional import (
    run_with_model,
    run_with_model_sync,
    run_agent_pipeline_sync,
    run_agent_pipeline,
    get_structured,
)
from llmling_agent.delegation import (
    AgentPool,
    ConversationController,
    Decision,
    EndDecision,
    RouteDecision,
    TalkBackDecision,
    interactive_controller,
    RuleBasedController,
)
from dotenv import load_dotenv
from llmling_agent.models.messages import ChatMessage
from llmling_agent.chat_session.base import AgentPoolView

__version__ = "0.17.0"

load_dotenv()

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentPool",
    "AgentPoolView",
    "AgentsManifest",
    "ChatMessage",
    "ConversationController",
    "Decision",
    "EndDecision",
    "HumanAgent",
    "RouteDecision",
    "RuleBasedController",
    "SystemPrompt",
    "TalkBackDecision",
    "get_structured",
    "interactive_controller",
    "run_agent_pipeline",
    "run_agent_pipeline_sync",
    "run_with_model",
    "run_with_model_sync",
]
