"""Agent configuration and creation."""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("llmling-agent")

from dotenv import load_dotenv

from llmling_agent.agent import Agent, AgentContext, AnyAgent, StructuredAgent
from llmling_agent.delegation import AgentPool, BaseTeam, Team, TeamRun
from llmling_agent.messaging.messagenode import MessageNode
from llmling_agent.messaging.messages import ChatMessage
from llmling_agent.models import AgentConfig, AgentsManifest
from llmling_agent.models.content import (
    AudioBase64Content,
    AudioURLContent,
    ImageBase64Content,
    ImageURLContent,
    PDFBase64Content,
    PDFURLContent,
    VideoURLContent,
)
from llmling_agent.tools import Tool, ToolCallInfo

load_dotenv()

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentContext",
    "AgentPool",
    "AgentsManifest",
    "AnyAgent",
    "AudioBase64Content",
    "AudioURLContent",
    "BaseTeam",
    "ChatMessage",
    "ImageBase64Content",
    "ImageURLContent",
    "MessageNode",
    "PDFBase64Content",
    "PDFURLContent",
    "StructuredAgent",
    "Team",
    "TeamRun",
    "Tool",
    "ToolCallInfo",
    "VideoURLContent",
    "__version__",
]
