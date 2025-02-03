from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from time import perf_counter
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
import tokonomics
from tokonomics import get_available_models
from tokonomics.toko_types import TokenUsage

from llmling_agent.common_types import ModelProtocol
from llmling_agent.log import get_logger
from llmling_agent.messaging.messages import ChatMessage, TokenCost
from llmling_agent.models.agents import ToolCallInfo
from llmling_agent.models.content import BaseContent, Content
from llmling_agent.tasks.exceptions import (
    ChainAbortedError,
    RunAbortedError,
    ToolSkippedError,
)
from llmling_agent.utils.inspection import has_argument_type
from llmling_agent_providers.base import (
    AgentLLMProvider,
    ProviderResponse,
    StreamingResponseProtocol,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from litellm import ChatCompletionMessageToolCall

    from llmling_agent.tools.base import ToolInfo


logger = get_logger(__name__)


@dataclass
class Usage:
    total_tokens: int | None
    request_tokens: int | None
    response_tokens: int | None


class LiteLLMModel:
    """Compatible model class for LiteLLM."""

    def __init__(self, model_name: str):
        self._name = model_name

    def name(self) -> str:
        return self._name.replace(":", "/")


@dataclass
class LiteLLMRunContext:
    """Simple run context for LiteLLM provider."""

    message_id: str
    model: LiteLLMModel
    prompt: str
    deps: Any


@dataclass
class LiteLLMStream[TResult]:
    """Wrapper to match StreamingResponseProtocol."""

    _stream: Any
    _response: Any | None = None
    model_name: str | None = None
    formatted_content: TResult | None = None
    is_complete: bool = False
    _accumulated_content: str = ""
    _final_usage: Usage | None = None

    async def stream(self) -> AsyncIterator[TResult]:
        """Stream chunks as they arrive."""
        try:
            final_chunk = None
            async for chunk in self._stream:
                if content := chunk.choices[0].delta.content:
                    self._accumulated_content += content
                    # Cast to expected type (usually str)
                    yield content  # type: ignore
                final_chunk = chunk

            self.is_complete = True
            self.formatted_content = self._accumulated_content  # type: ignore

            # Store usage from final chunk if available
            if final_chunk and hasattr(final_chunk, "usage"):
                self._final_usage = TokenUsage(
                    total_tokens=final_chunk.usage.total_tokens,  # type: ignore
                    request_tokens=final_chunk.usage.prompt_tokens,  # type: ignore
                    response_tokens=final_chunk.usage.completion_tokens,  # type: ignore
                )

        except Exception as e:
            logger.exception("Error during streaming")
            self.is_complete = True
            msg = "Streaming failed"
            raise RuntimeError(msg) from e

    def usage(self) -> Usage:
        """Get token usage statistics."""
        if not self._final_usage:
            return Usage(total_tokens=0, request_tokens=0, response_tokens=0)
        return self._final_usage


class LiteLLMProvider(AgentLLMProvider[Any]):
    """Provider using LiteLLM for model-agnostic completions."""

    NAME = "litellm"

    def __init__(
        self,
        name: str,
        model: str | ModelProtocol | None = None,
        *,
        debug: bool = False,
        model_settings: dict[str, Any] | None = None,
        retries: int = 1,
    ):
        super().__init__(name=name, debug=debug)
        self._model = model
        self.num_retries = retries
        self.model_settings = model_settings or {}

    async def get_model_names(self) -> list[str]:
        """Get list of all known model names."""
        return await get_available_models()

    async def handle_tool_call(
        self,
        tool_call: ChatCompletionMessageToolCall,
        tool: ToolInfo,
        message_id: str,
    ) -> tuple[ToolCallInfo, dict]:
        """Handle a single tool call properly."""
        function_args = json.loads(tool_call.function.arguments)
        original_tool = tool.callable.callable
        start_time = perf_counter()

        try:
            # 1. Handle confirmation if we have context
            if self._context:
                result = await self._context.handle_confirmation(
                    self._context, tool, function_args
                )
                match result:
                    case "skip":
                        msg = f"Tool {tool.name} execution skipped"
                        raise ToolSkippedError(msg)  # noqa: TRY301
                    case "abort_run":
                        msg = "Run aborted by user"
                        raise RunAbortedError(msg)  # noqa: TRY301
                    case "abort_chain":
                        msg = "Agent chain aborted by user"
                        raise ChainAbortedError(msg)  # noqa: TRY301
                    case "allow":
                        pass  # Continue with execution

            # 2. Add context if needed
            if has_argument_type(original_tool, "AgentContext"):
                enhanced_function_args = {"ctx": self._context, **function_args}
            else:
                enhanced_function_args = function_args
            # 3. Handle sync/async execution
            result = await tool.execute(**enhanced_function_args)
            # Create tool call info
            info = ToolCallInfo(
                tool_name=tool.name,
                agent_name=self.name,
                args=function_args,
                result=result,
                tool_call_id=tool_call.id,
                timing=perf_counter() - start_time,
                message_id=message_id,
            )
            self.tool_used.emit(info)
            message = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": str(result),
            }
        except (ToolSkippedError, RunAbortedError, ChainAbortedError) as e:
            # Handle confirmation-related errors
            info = ToolCallInfo(
                tool_name=tool.name,
                agent_name=self.name,
                args=function_args,
                result=str(e),
                tool_call_id=tool_call.id,
                error=str(e),
                message_id=message_id,
            )
            message = {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_call.function.name,
                "content": str(e),
            }
            return info, message
        else:
            return info, message

    async def generate_response(
        self,
        *prompts: str | Content,
        message_id: str,
        message_history: list[ChatMessage],
        result_type: type[Any] | None = None,
        model: ModelProtocol | str | None = None,
        tools: list[ToolInfo] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate response using LiteLLM."""
        from litellm import Choices, acompletion
        from litellm.files.main import ModelResponse

        model_name = self._get_model_name(model)
        try:
            # Create messages list from history and new prompt
            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for msg in message_history:
                messages.extend(self._convert_message_to_chat(msg))

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

            schemas = [t.schema for t in tools or []]
            # Get completion
            response = await acompletion(
                stream=False,
                model=model_name,
                messages=messages,
                response_format=result_type
                if result_type and issubclass(result_type, BaseModel)
                else None,
                num_retries=self.num_retries,
                tools=schemas or None,
                tool_choice="auto" if schemas else None,
                **self.model_settings,
            )
            assert isinstance(response, ModelResponse)
            assert isinstance(response.choices[0], Choices)
            calls: list[ToolCallInfo] = []
            new_messages = []
            if tool_calls := response.choices[0].message.tool_calls:
                pre = {"role": "assistant", "content": None, "tool_calls": tool_calls}
                new_messages.append(pre)
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    if not function_name:
                        continue
                    tool = self.tool_manager.get(function_name)
                    info, message = await self.handle_tool_call(
                        tool_call, tool, message_id
                    )
                    calls.append(info)
                    new_messages.append(message)
                import devtools

                devtools.debug(new_messages)
                response = await acompletion(
                    model=model_name,
                    messages=messages + new_messages,
                    stream=False,
                    **self.model_settings,
                )
                assert isinstance(response, ModelResponse)
                assert isinstance(response.choices[0], Choices)
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
                # Store in history if requested
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

    def _get_model_name(self, override: ModelProtocol | str | None = None) -> str:
        """Get effective model name."""
        if isinstance(override, ModelProtocol):
            return override.name()
        if isinstance(override, str):
            return override.replace(":", "/")
        if isinstance(self._model, ModelProtocol):
            return self._model.name()
        if self._model:
            return self._model.replace(":", "/")
        return "openai/gpt-4o-mini"

    def _convert_message_to_chat(self, message: Any) -> list[dict[str, str]]:
        """Convert message to chat format."""
        # This is a basic implementation - would need to properly handle
        # different message types and parts
        return [{"role": "user", "content": str(message)}]

    @asynccontextmanager
    async def stream_response(
        self,
        *prompts: str | Content,
        message_id: str,
        message_history: list[ChatMessage],
        result_type: type[Any] | None = None,
        model: ModelProtocol | str | None = None,
        tools: list[ToolInfo] | None = None,
        store_history: bool = True,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingResponseProtocol[Any]]:
        """Stream responses from LiteLLM."""
        from litellm import acompletion

        model_name = self._get_model_name(model)
        messages: list[dict[str, Any]] = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        if store_history:
            for msg in message_history:
                messages.extend(self._convert_message_to_chat(msg))

        # Convert new prompts to message content
        content_parts: list[dict[str, Any]] = []
        for p in prompts:
            match p:
                case str():
                    content_parts.append({"type": "text", "text": p})
                case BaseContent():
                    content_parts.append(p.to_openai_format())
        messages.append({"role": "user", "content": content_parts})
        schemas = [t.schema for t in tools or []]

        try:
            # Get streaming completion
            completion_stream = await acompletion(
                stream=True,
                model=model_name,
                messages=messages,
                response_format=result_type
                if result_type and issubclass(result_type, BaseModel)
                else None,
                tools=schemas,
                tool_choice="auto" if schemas else None,
                num_retries=self.num_retries,
                **self.model_settings,
            )

            # Create stream wrapper that matches our protocol
            stream = LiteLLMStream[Any](_stream=completion_stream, model_name=model_name)

            try:
                yield stream

                # Store in history if requested and stream completed
                if store_history and stream.is_complete:
                    request_msgs = [
                        ChatMessage(role="user", content=str(p)) for p in prompts
                    ]
                    content = stream.formatted_content
                    response_msg = ChatMessage(role="assistant", content=content)
                    self.conversation.add_chat_messages([*request_msgs, response_msg])

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
