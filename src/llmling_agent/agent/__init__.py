"""CLI commands for llmling-agent."""

from __future__ import annotations
from typing import Any, TypeVar

from llmling_agent.agent.agent import Agent
from llmling_agent.agent.agent_logger import AgentLogger
from llmling_agent.agent.conversation import ConversationManager
from llmling_agent.agent.human import HumanAgent
from llmling_agent.agent.structured import StructuredAgent
from llmling_agent.agent.slashed_agent import SlashedAgent


TDeps = TypeVar("TDeps")
AnyAgent = Agent[TDeps] | StructuredAgent[TDeps, Any]


__all__ = [
    "Agent",
    "AgentLogger",
    "AnyAgent",
    "ConversationManager",
    "HumanAgent",
    "SlashedAgent",
    "StructuredAgent",
]
