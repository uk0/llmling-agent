"""LiteLLM Provider."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
import tokonomics
from tokonomics import get_available_models
from tokonomics.toko_types import TokenUsage

from llmling_agent.common_types import ModelProtocol
from llmling_agent.log import get_logger
from llmling_agent.messaging.messages import ChatMessage, TokenCost
from llmling_agent.models.content import BaseContent, Content
from llmling_agent_providers.base import (
    AgentLLMProvider,
    ProviderResponse,
    StreamingResponseProtocol,
    UsageLimits,
)
from llmling_agent_providers.litellm_provider.call_wrapper import FakeAgent
from llmling_agent_providers.litellm_provider.stream import LiteLLMStream
from llmling_agent_providers.tool_call_handler import ToolCallHandler
from llmling_agent_providers.utils import convert_message_to_chat


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from litellm import ChatCompletionMessageToolCall

    from llmling_agent.agent.context import AgentContext
    from llmling_agent.tools import ToolCallInfo
    from llmling_agent.tools.base import Tool


logger = get_logger(__name__)


def convert_litellm_tool_call(tool_call: ChatCompletionMessageToolCall) -> dict[str, Any]:
    """Convert LiteLLM tool call to standard format."""
    return {
        "id": tool_call.id,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
        "type": tool_call.type,
    }


class LiteLLMProvider(AgentLLMProvider[Any]):
    """Provider using LiteLLM for model-agnostic completions."""

    NAME = "litellm"

    def __init__(
        self,
        name: str,
        model: str | ModelProtocol | None = None,
        context: AgentContext | None = None,
        *,
        debug: bool = False,
        model_settings: dict[str, Any] | None = None,
        retries: int = 1,
    ):
        super().__init__(
            name=name,
            debug=debug,
            model_settings=model_settings,
            context=context,
        )
        self._model = model
        self.num_retries = retries

    async def get_model_names(self) -> list[str]:
        """Get list of all known model names."""
        return await get_available_models()

    async def generate_response(
        self,
        *prompts: str | Content,
        message_id: str,
        message_history: list[ChatMessage],
        result_type: type[Any] | None = None,
        model: ModelProtocol | str | None = None,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        usage_limits: UsageLimits | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate response using LiteLLM."""
        from litellm import Choices
        from litellm.files.main import ModelResponse

        tools = tools or []
        model_name = self._get_model_name(model)
        agent = FakeAgent(model_name, model_settings=self.model_settings)
        tool_handler = ToolCallHandler(self.name)
        tool_handler.tool_used.connect(self.tool_used)

        try:
            # Create messages list from history and new prompt
            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for msg in message_history:
                messages.extend(convert_message_to_chat(msg))

            # Convert new prompts to message content
            content_parts: list[dict[str, Any]] = []
            for p in prompts:
                match p:
                    case str():
                        content_parts.append({"type": "text", "text": p})
                    case BaseContent():
                        content_parts.append(p.to_openai_format())
            # Add the multi-modal content as user message
            messages.append({"role": "user", "content": content_parts})
            # Get completion
            response = await agent.run(
                messages=messages,
                usage_limits=usage_limits,
                result_type=result_type,
                num_retries=self.num_retries,
                tools=tools,
            )
            assert isinstance(response, ModelResponse)
            assert isinstance(response.choices[0], Choices)
            calls: list[ToolCallInfo] = []
            if tool_calls := response.choices[0].message.tool_calls:
                raw_tool_calls = [convert_litellm_tool_call(tc) for tc in tool_calls]
                new_messages, calls = await tool_handler.handle_tool_calls(
                    raw_tool_calls,
                    tools,
                    self._context,
                    message_id,
                )
                response = await agent.run(messages=messages + new_messages)
            # Extract content
            content: Any = response.choices[0].message.content  # type: ignore
            if content and result_type and issubclass(result_type, BaseModel):
                # Parse JSON string into the requested model
                content = result_type.model_validate_json(content)
            # Create tokonomics usage
            usage = TokenUsage(
                total_tokens=response.usage.prompt_tokens,  # type: ignore
                request_tokens=response.usage.prompt_tokens,  # type: ignore
                response_tokens=response.usage.completion_tokens,  # type: ignore
            )
            try:
                cost_and_usage = TokenCost(
                    token_usage=usage,
                    total_cost=response.usage.cost,  # type: ignore
                )
            except Exception:  # noqa: BLE001
                cost = await tokonomics.calculate_token_cost(
                    model_name,
                    completion_tokens=response.usage.completion_tokens,  # type: ignore
                    prompt_tokens=response.usage.prompt_tokens,  # type: ignore
                )
                if cost:
                    cost_and_usage = TokenCost(
                        token_usage=usage,
                        total_cost=cost.total_cost,  # type: ignore
                    )
                else:
                    cost_and_usage = None

            return ProviderResponse(
                content=content,
                tool_calls=calls,
                model_name=model_name,
                cost_and_usage=cost_and_usage,
                provider_extra=response.choices[0].message.provider_specific_fields,  # type: ignore
            )

        except Exception as e:
            logger.exception("LiteLLM completion failed")
            error_msg = f"LiteLLM completion failed: {e}"
            raise RuntimeError(error_msg) from e
        finally:
            tool_handler.tool_used.disconnect(self.tool_used)

    def _get_model_name(self, override: ModelProtocol | str | None = None) -> str:
        """Get effective model name."""
        final_model = override or self._model
        match final_model:
            case ModelProtocol():
                name = final_model.model_name
            case str():
                name = final_model
            case _:
                msg = "No model specified"
                raise ValueError(msg)
        name = name.replace(":", "/")
        if "/" in name:
            return name
        if name.startswith("gpt"):
            return f"openai/{name}"
        if name.startswith("claude"):
            return f"anthropic/{name}"
        return name

    @asynccontextmanager
    async def stream_response(
        self,
        *prompts: str | Content,
        message_id: str,
        message_history: list[ChatMessage],
        result_type: type[Any] | None = None,
        model: ModelProtocol | str | None = None,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        usage_limits: UsageLimits | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingResponseProtocol[Any]]:
        """Stream responses from LiteLLM."""
        model_name = self._get_model_name(model)
        agent = FakeAgent(model_name, model_settings=self.model_settings)
        messages: list[dict[str, Any]] = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for msg in message_history:
            messages.extend(convert_message_to_chat(msg))

        # Convert new prompts to message content
        content_parts: list[dict[str, Any]] = []
        for p in prompts:
            match p:
                case str():
                    content_parts.append({"type": "text", "text": p})
                case BaseContent():
                    content_parts.append(p.to_openai_format())
        messages.append({"role": "user", "content": content_parts})

        try:
            # Get streaming completion
            completion_stream = await agent.run_stream(
                messages=messages,
                usage_limits=usage_limits,
                result_type=result_type,
                tools=tools,
                num_retries=self.num_retries,
            )

            # Create stream wrapper that matches our protocol
            stream = LiteLLMStream[Any](_stream=completion_stream, model_name=model_name)

            try:
                yield stream
            except Exception as e:
                logger.exception("Error during stream processing")
                error_msg = "Stream processing failed"
                raise RuntimeError(error_msg) from e

        except Exception as e:
            logger.exception("Failed to create stream")
            error_msg = "Stream creation failed"
            raise RuntimeError(error_msg) from e


if __name__ == "__main__":
    import asyncio
    import logging

    from llmling_agent import Agent

    logging.basicConfig(level=logging.INFO)

    async def main():
        # Create agent with LiteLLM provider
        agent = Agent[Any](provider="litellm", model="openai/gpt-3.5-turbo", name="test")

        # Test normal completion
        print("\nNormal completion:")
        response = await agent.run("Tell me a short joke about Python programming.")
        print(f"Response from {agent.model_name}:")
        print(f"Content: {response.content}")

        # Test streaming
        print("\nStreaming completion:")
        async with agent.run_stream("Write a haiku about coding.") as stream:
            print("Streaming chunks:")
            async for chunk in stream.stream():
                print(chunk, end="", flush=True)
            print("\n")
            print(f"Final content: {stream.formatted_content}")
            usage = stream.usage()
            print(f"Total tokens: {usage.total_tokens}")
            print(f"Model used: {stream.model_name}")

    asyncio.run(main())
