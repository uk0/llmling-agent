from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import pathlib
from typing import TYPE_CHECKING

from platformdirs import user_data_dir


if TYPE_CHECKING:
    from llmling_agent.chat_session.models import ChatMessage

HISTORY_DIR = pathlib.Path(user_data_dir("llmling", "llmling")) / "history"


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

    def update_tokens(self, message: ChatMessage) -> None:
        """Update token counts from message metadata."""
        if message.metadata and (token_usage := message.metadata.get("token_usage")):
            self.total_tokens += token_usage.get("total", 0)
            self.prompt_tokens += token_usage.get("prompt", 0)
            self.completion_tokens += token_usage.get("completion", 0)

        if message.metadata and (cost := message.metadata.get("cost")):
            self.total_cost += cost
