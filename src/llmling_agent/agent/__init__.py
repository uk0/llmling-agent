"""CLI commands for llmling-agent."""

from __future__ import annotations
from typing_extensions import TypeVar

from llmling_agent.agent.agent import Agent
from llmling_agent.agent.agent_logger import AgentLogger
from llmling_agent.agent.conversation import ConversationManager
from llmling_agent.agent.container import AgentContainer
from llmling_agent.agent.structured import StructuredAgent
from llmling_agent.agent.slashed_agent import SlashedAgent


TDeps = TypeVar("TDeps")
TResult = TypeVar("TResult", default=str)


type AnyAgent[TDeps, TResult] = Agent[TDeps] | StructuredAgent[TDeps, TResult]


__all__ = [
    "Agent",
    "AgentContainer",
    "AgentLogger",
    "AnyAgent",
    "ConversationManager",
    "SlashedAgent",
    "StructuredAgent",
]
