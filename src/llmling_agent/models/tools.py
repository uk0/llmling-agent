"""Models for tools."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ToolCallInfo(BaseModel):
    """Information about an executed tool call."""

    tool_name: str
    """Name of the tool that was called."""

    args: dict[str, Any]
    """Arguments passed to the tool."""

    result: Any
    """Result returned by the tool."""

    agent_name: str
    """Name of the calling agent."""

    tool_call_id: str = Field(default_factory=lambda: str(uuid4()))
    """ID provided by the model (e.g. OpenAI function call ID)."""

    timestamp: datetime = Field(default_factory=datetime.now)
    """When the tool was called."""

    message_id: str | None = None
    """ID of the message that triggered this tool call."""

    context_data: Any | None = None
    """Optional context data that was passed to the agent's run() method."""

    error: str | None = None
    """Error message if the tool call failed."""

    timing: float | None = None
    """Time taken for this specific tool call in seconds."""

    agent_tool_name: str | None = None
    """If this tool is agent-based, the name of that agent."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
