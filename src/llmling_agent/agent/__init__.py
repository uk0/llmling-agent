"""CLI commands for llmling-agent."""

from __future__ import annotations
from typing_extensions import TypeVar

from llmling_agent.agent.agent import Agent
from llmling_agent.agent.structured import StructuredAgent

type AnyAgent[TDeps, TResult] = Agent[TDeps] | StructuredAgent[TDeps, TResult]

from llmling_agent.agent.agent_logger import AgentLogger
from llmling_agent.agent.conversation import ConversationManager
from llmling_agent.agent.container import AgentContainer
from llmling_agent.agent.talk import Interactions


TDeps = TypeVar("TDeps")
TResult = TypeVar("TResult", default=str)


__all__ = [
    "Agent",
    "AgentContainer",
    "AgentLogger",
    "AnyAgent",
    "ConversationManager",
    "Interactions",
    "StructuredAgent",
]
