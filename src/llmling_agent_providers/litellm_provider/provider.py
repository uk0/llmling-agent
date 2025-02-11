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

from llmling_agent.agent.context import AgentContext
from llmling_agent.common_types import ModelProtocol
from llmling_agent.log import get_logger
from llmling_agent.messaging.messages import ChatMessage, TokenCost
from llmling_agent.models.content import BaseContent, Content
from llmling_agent.models.tools import ToolCallInfo
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
    UsageLimits,
)
from llmling_agent_providers.litellm_provider.call_wrapper import FakeAgent
from llmling_agent_providers.litellm_provider.utils import Usage, convert_message_to_chat


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from litellm import ChatCompletionMessageToolCall

    from llmling_agent.tools.base import ToolInfo


logger = get_logger(__name__)


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
                result = await self._context.handle_confirmation(tool, function_args)
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
            if has_argument_type(original_tool, AgentContext):
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
                context_data=self._context.data if self._context else None,
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
                context_data=self._context.data if self._context else None,
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

    async def handle_tool_calls(
        self,
        tool_calls: list[ChatCompletionMessageToolCall],
        tools: list[ToolInfo],
        message_id: str,
    ) -> tuple[list[dict[str, Any]], list[ToolCallInfo]]:
        calls: list[ToolCallInfo] = []
        new_messages = []
        pre = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        new_messages.append(pre)
        for i, tool_call in enumerate(tool_calls):
            if self._context and self._context.report_progress:
                await self._context.report_progress(i, None)
            function_name = tool_call.function.name
            if not function_name:
                continue
            tool = next(i for i in tools if i.name == function_name)
            info, message = await self.handle_tool_call(tool_call, tool, message_id)
            calls.append(info)
            new_messages.append(message)
        return new_messages, calls

    async def generate_response(
        self,
        *prompts: str | Content,
        message_id: str,
        message_history: list[ChatMessage],
        result_type: type[Any] | None = None,
        model: ModelProtocol | str | None = None,
        tools: list[ToolInfo] | None = None,
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
                new_messages, calls = await self.handle_tool_calls(
                    tool_calls,
                    tools,
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
        tools: list[ToolInfo] | None = None,
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
