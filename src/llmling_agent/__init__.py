"""Agent configuration and creation."""

from llmling_agent.models import AgentsManifest, SystemPrompt, AgentConfig
from llmling_agent.agent import Agent, StructuredAgent, SlashedAgent
from llmling_agent.functional import (
    run_with_model,
    run_with_model_sync,
    run_agent_pipeline_sync,
    run_agent_pipeline,
    get_structured,
)
from llmling_agent.delegation import (
    AgentPool,
    Decision,
    EndDecision,
    RouteDecision,
    AwaitResponseDecision,
    interactive_controller,
)
from llmling_agent.delegation.callbacks import DecisionCallback
from llmling_agent.delegation.router import AgentRouter, RuleRouter, CallbackRouter
from dotenv import load_dotenv
from llmling_agent.models.messages import ChatMessage
from llmling_agent.chat_session.base import AgentPoolView

__version__ = "0.20.0"

load_dotenv()

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentPool",
    "AgentPoolView",
    "AgentRouter",
    "AgentsManifest",
    "AwaitResponseDecision",
    "CallbackRouter",
    "ChatMessage",
    "Decision",
    "DecisionCallback",
    "EndDecision",
    "RouteDecision",
    "RuleRouter",
    "SlashedAgent",
    "StructuredAgent",
    "SystemPrompt",
    "get_structured",
    "interactive_controller",
    "run_agent_pipeline",
    "run_agent_pipeline_sync",
    "run_with_model",
    "run_with_model_sync",
]
