"""Agent configuration and creation."""

from llmling_agent.models import AgentsManifest, AgentConfig
from llmling_agent.agent import Agent, StructuredAgent, AnyAgent, AgentContext

from llmling_agent.common_types import PythonCode, TOMLCode, JSONCode, YAMLCode
from llmling_agent.delegation import AgentPool, Team, TeamRun, BaseTeam
from dotenv import load_dotenv
from llmling_agent.messaging.messages import ChatMessage
from llmling_agent.tools import Tool, ToolCallInfo
from llmling_agent.messaging.messagenode import MessageNode
from llmling_agent.models.content import (
    PDFURLContent,
    PDFBase64Content,
    ImageBase64Content,
    ImageURLContent,
    AudioURLContent,
    AudioBase64Content,
    VideoURLContent,
)

__version__ = "0.99.33"

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
    "JSONCode",
    "MessageNode",
    "PDFBase64Content",
    "PDFURLContent",
    "PythonCode",
    "StructuredAgent",
    "TOMLCode",
    "Team",
    "TeamRun",
    "Tool",
    "ToolCallInfo",
    "VideoURLContent",
    "YAMLCode",
]
