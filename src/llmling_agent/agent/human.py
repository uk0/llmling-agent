from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from typing_extensions import TypeVar

from llmling_agent.agent.agent import Agent
from llmling_agent.models.messages import ChatMessage


if TYPE_CHECKING:
    from pydantic_ai.agent import models
    from pydantic_ai.result import Usage


TResult = TypeVar("TResult", default=str)


class HumanAgent[TDeps](Agent[TDeps]):
    """Agent implementation for human-in-the-loop interaction.

    Simplifies the Agent interface to just handle basic request/response
    without system prompts, tools, or retry logic.
    """

    async def run(
        self,
        *prompt: str,
        result_type: type[TResult] | None = None,
        deps: TDeps | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        usage: Usage | None = None,
    ) -> ChatMessage[TResult]:
        """Get human input in response to prompt.

        Args:
            prompt: Prompt(s) to show to human
            result_type: Optional type for structured responses
            deps: Optional dependencies for the agent
            message_history: Previous messages (maintained for interface compatibility)
            model: Not used for human agent
            usage: Not used for human agent

        Returns:
            Human's response as a ChatMessage
        """
        final_prompt = "\n\n".join(prompt)

        # Get human input
        print(f"\n{final_prompt}")
        response = input("> ")

        # Create pydantic-ai messages for conversation history
        request = ModelRequest(parts=[UserPromptPart(content=final_prompt)])
        response_msg = ModelResponse(parts=[TextPart(content=response)])

        # Update conversation history
        self.conversation._last_messages = [request, response_msg]
        if not message_history:  # Only update if not in middle of chain
            self.conversation.set_history([
                *self.conversation.get_history(),
                request,
                response_msg,
            ])

        # Return standard ChatMessage
        # TODO: get this type issue in order
        return ChatMessage(  # type: ignore
            content=response,
            role="user",
            name=self.name or "human",
            message_id=str(uuid4()),
        )
