"""Logging functionality for agent interactions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from llmling_agent.storage import Conversation, Message
from llmling_agent.storage.models import ToolCall


if TYPE_CHECKING:
    from llmling_agent import LLMlingAgent
    from llmling_agent.models.agents import ToolCallInfo
    from llmling_agent.models.messages import ChatMessage


logger = logging.getLogger(__name__)


class AgentLogger:
    """Handles database logging for agent interactions."""

    def __init__(self, agent: LLMlingAgent[Any, Any], enable_logging: bool = True):
        """Initialize logger.

        Args:
            agent: Agent to log interactions for
            enable_logging: Whether to enable logging
        """
        self.agent = agent
        self.enable_logging = enable_logging
        self.conversation_id = str(uuid4())
        self.message_history: list[ChatMessage] = []
        self.toolcall_history: list[ToolCallInfo] = []

        # Initialize conversation record if enabled
        if enable_logging:
            self.init_conversation()
            # Connect to the combined signal to capture all messages
            agent.message_exchanged.connect(self.log_message)
            agent.tool_used.connect(self.log_tool_call)

    def clear_state(self):
        """Clear agent state."""
        self.message_history.clear()
        self.toolcall_history.clear()

    @property
    def last_message(self) -> ChatMessage[Any] | None:
        """Get last message in history."""
        return self.message_history[-1] if self.message_history else None

    @property
    def last_tool_call(self) -> ToolCallInfo | None:
        """Get last tool call in history."""
        return self.toolcall_history[-1] if self.toolcall_history else None

    def init_conversation(self):
        """Create initial conversation record."""
        Conversation.log(self.conversation_id, self.agent.name)

    def log_message(self, message: ChatMessage):
        """Handle message from chat signal."""
        self.message_history.append(message)

        if not self.enable_logging:
            return

        Message.log(
            conversation_id=self.conversation_id,
            content=str(message.content),
            role=message.role,
            name=message.name,
            cost_info=message.cost_info,
            model=message.model,
            response_time=message.response_time,
        )

    def log_tool_call(self, tool_call: ToolCallInfo):
        """Handle tool usage signal."""
        self.toolcall_history.append(tool_call)

        if not self.enable_logging:
            return

        ToolCall.log(
            conversation_id=self.conversation_id,
            message_id=tool_call.message_id or "",
            tool_call=tool_call,
        )
