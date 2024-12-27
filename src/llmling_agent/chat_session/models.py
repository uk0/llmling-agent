"""Data models for agent chat sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.models.messages import ChatMessage


logger = get_logger(__name__)


class ChatSessionMetadata(BaseModel):
    """Configuration and state of an agent chat session."""

    session_id: str
    agent_name: str
    model: str | None = None
    tool_states: dict[str, bool]
    start_time: datetime = Field(default_factory=datetime.now)


@dataclass
class SessionState:
    """Current state of the interactive session."""

    current_model: str | None = None
    message_count: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    last_command: str | None = None

    def update_tokens(self, message: ChatMessage):
        """Update token counts and costs from message metadata."""
        if not message.metadata:
            return
        if token_usage := message.metadata.token_usage:
            self.total_tokens = token_usage["total"]
            self.prompt_tokens = token_usage["prompt"]
            self.completion_tokens = token_usage["completion"]

        if cost := message.metadata.cost:
            self.total_cost = float(cost)
            logger.debug("Updated session cost to: $%.6f", self.total_cost)

    @property
    def duration(self) -> str:
        """Get formatted duration since session start."""
        duration = datetime.now() - self.start_time
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
