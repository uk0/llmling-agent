"""Agent provider implementations."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from llmling import ToolError
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.agent import EndStrategy, models
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from llmling_agent.log import get_logger
from llmling_agent.models.context import AgentContext
from llmling_agent.models.messages import ChatMessage, TokenAndCostResult
from llmling_agent.pydantic_ai_utils import extract_usage, format_response


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from pydantic_ai.messages import ModelMessage

    from llmling_agent.common_types import ToolType


logger = get_logger(__name__)


@runtime_checkable
class AgentProvider(Protocol):
    """Protocol for agent response generation."""

    async def generate_response(
        self,
        prompt: str,
        *,
        result_type: type[Any] | None = None,
        deps: Any | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> ChatMessage[Any]:
        """Generate a response for the given prompt.

        Args:
            prompt: Text prompt to respond to
            result_type: Optional type for structured responses
            deps: Optional dependency injection data
            message_history: Optional previous messages for context
            model: Optional model override

        Returns:
            Response message with optional structured content
        """
        ...

    def stream_response(
        self,
        prompt: str,
        *,
        result_type: type[Any] | None = None,
        deps: Any | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> AsyncIterator[ChatMessage[Any]]: ...


class PydanticAIProvider(AgentProvider):
    """Provider using pydantic-ai as backend."""

    def __init__(
        self,
        *,
        model: str | models.Model | None = None,
        system_prompt: str | Sequence[str] = (),
        tools: Sequence[ToolType] | None = None,
        retries: int = 1,
        result_retries: int | None = None,
        end_strategy: EndStrategy = "early",
        defer_model_check: bool = False,
        context: AgentContext[Any] | None = None,
    ):
        """Initialize pydantic-ai backend.

        Args:
            model: Model to use for responses
            system_prompt: Initial system instructions
            tools: Available tools
            retries: Number of retries for failed operations
            result_retries: Max retries for result validation
            end_strategy: How to handle tool calls with final result
            defer_model_check: Whether to defer model validation
            context: Optional agent context
        """
        if isinstance(model, str):
            model = models.infer_model(model)  # type: ignore

        self._model = model
        self._agent = PydanticAgent(
            model=model,  # type: ignore
            system_prompt=system_prompt,
            tools=[],
            retries=retries,
            end_strategy=end_strategy,
            result_retries=result_retries,
            defer_model_check=defer_model_check,
            deps_type=AgentContext,
        )
        self._context = context

    async def generate_response(
        self,
        prompt: str,
        *,
        result_type: type[Any] | None = None,
        deps: Any | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> ChatMessage[Any]:
        """Generate response using pydantic-ai.

        Args:
            prompt: Text prompt to respond to
            result_type: Optional type for structured responses
            deps: Optional dependency injection data
            message_history: Optional previous messages for context
            model: Optional model override

        Returns:
            Response message with optional structured content
        """
        if deps is not None and self._context is not None:
            self._context.data = deps

        start_time = datetime.now()

        try:
            # Run through pydantic-ai
            result = await self._agent.run(
                prompt,
                deps=self._context,  # type: ignore
                message_history=message_history,
                model=model,
            )

            # Extract cost info
            usage = result.usage()
            cost_info: TokenAndCostResult | None = None
            if usage:
                model_str = str(self._model) if self._model else None
                if model_str:
                    cost_info = await extract_usage(
                        usage,
                        model_str,
                        prompt,
                        str(result.data),
                    )

            # Create chat message
            return ChatMessage[Any](
                content=result.data,
                role="assistant",
                name=self._context.agent_name if self._context else None,
                model=str(self._model) if self._model else None,
                message_id=str(uuid4()),
                cost_info=cost_info,
                response_time=(datetime.now() - start_time).total_seconds(),
            )

        except Exception as e:
            logger.exception("Error generating response")
            msg = f"Response generation failed: {e}"
            raise ToolError(msg) from e

    async def stream_response(
        self,
        prompt: str,
        *,
        result_type: type[Any] | None = None,
        deps: Any | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> AsyncIterator[ChatMessage[Any]]:
        """Stream response using pydantic-ai."""
        if deps is not None and self._context is not None:
            self._context.data = deps

        start_time = datetime.now()

        try:
            async with self._agent.run_stream(
                prompt,
                deps=self._context,  # type: ignore
                message_history=message_history,
                model=model,
            ) as stream_result:
                # Stream intermediate chunks
                async for response in stream_result.stream():
                    yield ChatMessage[Any](
                        content=str(response),
                        role="assistant",
                        name=self._context.agent_name if self._context else None,
                        model=str(self._model) if self._model else None,
                    )

                # Once stream is complete, we can get metrics
                if stream_result.is_complete:
                    # Get usage info if available
                    usage = stream_result.usage()
                    cost_info: TokenAndCostResult | None = None
                    if usage:
                        model_str = str(self._model) if self._model else None
                        if model_str:
                            # Get final response from messages
                            messages = stream_result.new_messages()
                            response_parts: list[str] = []
                            for msg in messages:
                                response_parts.extend(
                                    format_response(part) for part in msg.parts
                                )

                            response_text = "\n".join(response_parts)
                            cost_info = await extract_usage(
                                usage,
                                model_str,
                                prompt,
                                response_text,
                            )

                    # Send final status message
                    yield ChatMessage[Any](
                        content="",  # Empty content for status message
                        role="assistant",
                        name=self._context.agent_name if self._context else None,
                        model=str(self._model) if self._model else None,
                        message_id=str(uuid4()),
                        cost_info=cost_info,
                        response_time=(datetime.now() - start_time).total_seconds(),
                    )

        except Exception as e:
            logger.exception("Error streaming response")
            msg = f"Response streaming failed: {e}"
            raise ToolError(msg) from e


class HumanProvider(AgentProvider):
    """Provider for human-in-the-loop responses."""

    def __init__(self, name: str | None = None):
        """Initialize human provider.

        Args:
            name: Optional name for the human agent
        """
        self.name = name or "human"

    async def generate_response(
        self,
        prompt: str,
        *,
        result_type: type[Any] | None = None,
        deps: Any | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> ChatMessage[Any]:
        """Get response through human input.

        Args:
            prompt: Text prompt to respond to
            result_type: Optional type for structured responses
            deps: Optional dependency injection data
            message_history: Optional previous messages for context
            model: Not used for human provider

        Returns:
            Response message from human input

        Note:
            If result_type is provided, will attempt to parse/validate input
            as that type. Otherwise returns raw text response.
        """
        start_time = datetime.now()

        # Show context if available
        if message_history:
            print("\nContext:")
            for msg in message_history:
                if isinstance(msg, ModelRequest):
                    parts = [p for p in msg.parts if isinstance(p, UserPromptPart)]
                    for part in parts:
                        print(f"User: {part.content}")
                elif isinstance(msg, ModelResponse):
                    parts = [p for p in msg.parts if isinstance(p, TextPart)]
                    for part in parts:
                        print(f"Assistant: {part.content}")
            print("\n---")

        # Show prompt and get response
        print(f"\n{prompt}")
        if result_type:
            print(f"(Please provide response as {result_type.__name__})")
        response = input("> ")

        # Parse structured response if needed
        content: Any = response
        if result_type:
            try:
                content = result_type.model_validate_json(response)
            except Exception as e:
                logger.exception("Failed to parse structured response")
                msg = f"Invalid response format: {e}"
                raise ToolError(msg) from e

        # Create chat message
        return ChatMessage[Any](
            content=content,
            role="user",
            name=self.name,
            message_id=str(uuid4()),
            response_time=(datetime.now() - start_time).total_seconds(),
        )
