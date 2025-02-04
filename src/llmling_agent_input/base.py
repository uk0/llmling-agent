from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine

    from llmling_agent.agent.context import AgentContext, ConfirmationResult
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.tools.base import ToolInfo


class InputProvider(ABC):
    """Base class for handling all UI interactions."""

    @abstractmethod
    def get_input(
        self,
        context: AgentContext,
        prompt: str,
        result_type: type | None = None,
        message_history: list[ChatMessage] | None = None,
    ) -> Coroutine[Any, Any, str]:
        """Get normal input (used by HumanProvider).

        Args:
            context: Current agent context
            prompt: The prompt to show to the user
            result_type: Optional type for structured responses
            message_history: Optional conversation history
        """

    async def get_streaming_input(
        self,
        context: AgentContext,
        prompt: str,
        result_type: type | None = None,
        message_history: list[ChatMessage] | None = None,
    ) -> AsyncIterator[str]:
        """Get streaming input (used by HumanProvider streaming mode).

        Args:
            context: Current agent context
            prompt: The prompt to show to the user
            result_type: Optional type for structured responses
            message_history: Optional conversation history
        """
        response = await self.get_input(context, prompt, result_type, message_history)
        yield response

    @abstractmethod
    def get_tool_confirmation(
        self,
        context: AgentContext,
        tool: ToolInfo,
        args: dict[str, Any],
        message_history: list[ChatMessage] | None = None,
    ) -> Coroutine[Any, Any, ConfirmationResult]:
        """Get tool execution confirmation.

        Args:
            context: Current agent context
            tool: Information about the tool to be executed
            args: Tool arguments
            message_history: Optional conversation history
        """

    @abstractmethod
    def get_code_input(
        self,
        context: AgentContext,
        template: str | None = None,
        language: str = "python",
        description: str | None = None,
    ) -> Coroutine[Any, Any, str]:
        """Get multi-line code input.

        Args:
            context: Current agent context
            template: Optional template code
            language: Programming language (for syntax highlighting)
            description: Optional description of expected code
        """
