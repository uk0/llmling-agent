"""CLI commands for llmling-agent."""

from __future__ import annotations
from typing_extensions import TypeVar

from llmling_agent.agent.agent import Agent
from llmling_agent.agent.structured import StructuredAgent
from llmling_agent.agent.context import AgentContext


TDeps = TypeVar("TDeps", default=None)
TResult = TypeVar("TResult", default=str)

type AnyAgent[TDeps, TResult] = Agent[TDeps] | StructuredAgent[TDeps, TResult]

from llmling_agent.agent.conversation import ConversationManager
from llmling_agent.agent.interactions import Interactions
from llmling_agent.agent.sys_prompts import SystemPrompts


__all__ = [
    "Agent",
    "AgentContext",
    "AnyAgent",
    "ConversationManager",
    "Interactions",
    "StructuredAgent",
    "SystemPrompts",
]
