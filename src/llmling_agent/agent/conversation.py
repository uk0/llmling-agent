"""Conversation management for LLMling agent."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

from llmling.prompts import BasePrompt, PromptMessage, StaticPrompt

from llmling_agent.log import get_logger
from llmling_agent.pydantic_ai_utils import convert_model_message


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from pydantic_ai.messages import ModelMessage

    from llmling_agent.agent.agent import LLMlingAgent
    from llmling_agent.models.messages import ChatMessage

logger = get_logger(__name__)

OverrideMode = Literal["replace", "append"]
type PromptInput = str | BasePrompt


def _to_base_prompt(prompt: PromptInput) -> BasePrompt:
    """Convert input to BasePrompt instance."""
    if isinstance(prompt, str):
        return StaticPrompt(
            name="System prompt",
            description="System prompt",
            messages=[PromptMessage(role="system", content=prompt)],
        )
    return prompt


class ConversationManager:
    """Manages conversation state and system prompts."""

    def __init__(
        self,
        agent: LLMlingAgent[Any, Any],
        initial_prompts: str | Sequence[str] | None = None,
    ):
        """Initialize conversation manager.

        Args:
            agent: instance to manage
            initial_prompts: Initial system prompts that start each conversation
        """
        self._agent = agent._pydantic_agent
        self._initial_prompts: list[BasePrompt] = []
        self._current_history: list[ModelMessage] | None = None
        self._last_messages: list[ModelMessage] = []

        # Add initial prompts
        if initial_prompts is not None:
            prompts_list = (
                [initial_prompts] if isinstance(initial_prompts, str) else initial_prompts
            )
            for prompt in prompts_list:
                self._initial_prompts.append(
                    StaticPrompt(
                        name="Initial system prompt",
                        description="Initial system prompt",
                        messages=[PromptMessage(role="system", content=prompt)],
                    )
                )

    @asynccontextmanager
    async def temporary(
        self,
        *,
        sys_prompts: PromptInput | Sequence[PromptInput] | None = None,
        mode: OverrideMode = "append",
    ) -> AsyncIterator[None]:
        """Start temporary conversation with different system prompts."""
        # Store original state
        original_prompts = list(self._initial_prompts)
        original_system_prompts = self._agent._system_prompts  # Store pydantic-ai prompts
        original_history = self._current_history

        try:
            if sys_prompts is not None:
                # Convert to list of BasePrompt
                new_prompts: list[BasePrompt] = []
                if isinstance(sys_prompts, str | BasePrompt):
                    new_prompts = [_to_base_prompt(sys_prompts)]
                else:
                    new_prompts = [_to_base_prompt(prompt) for prompt in sys_prompts]

                self._initial_prompts = (
                    original_prompts + new_prompts if mode == "append" else new_prompts
                )

                # Update pydantic-ai's system prompts
                formatted_prompts = await self.get_all_prompts()
                self._agent._system_prompts = tuple(formatted_prompts)

            # Force new conversation
            self._current_history = None
            yield
        finally:
            # Restore complete original state
            self._initial_prompts = original_prompts
            self._agent._system_prompts = original_system_prompts
            self._current_history = original_history

    def add_prompt(self, prompt: PromptInput):
        """Add a system prompt.

        Args:
            prompt: String content or BasePrompt instance to add
        """
        self._initial_prompts.append(_to_base_prompt(prompt))

    async def get_all_prompts(self) -> list[str]:
        """Get all formatted system prompts in order."""
        result: list[str] = []

        for prompt in self._initial_prompts:
            try:
                messages = await prompt.format()
                result.extend(
                    msg.get_text_content() for msg in messages if msg.role == "system"
                )
            except Exception:
                logger.exception("Error formatting prompt")

        return result

    def get_history(self) -> list[ModelMessage] | None:
        """Get current conversation history."""
        return self._current_history

    def set_history(self, history: list[ModelMessage]):
        """Update conversation history after run."""
        self._current_history = history

    def clear(self):
        """Clear conversation history and prompts."""
        self._initial_prompts.clear()
        self._current_history = None
        self._last_messages = []

    @property
    def last_run_messages(self) -> list[ChatMessage]:
        """Get messages from the last run converted to our format."""
        return [convert_model_message(msg) for msg in self._last_messages]
