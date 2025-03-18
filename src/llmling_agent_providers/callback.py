"""Callable-based provider."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import inspect
from typing import TYPE_CHECKING, Any, TypeVar

from llmling_agent.agent.context import AgentContext
from llmling_agent.log import get_logger
from llmling_agent.models.content import BaseContent
from llmling_agent.prompts.convert import format_prompts
from llmling_agent.utils.inspection import execute, has_argument_type
from llmling_agent_providers.base import (
    AgentProvider,
    ProviderResponse,
    StreamingResponseProtocol,
    UsageLimits,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.models.content import Content
    from llmling_agent_config.providers import ProcessorCallback


logger = get_logger(__name__)

TResult = TypeVar("TResult")


class CallbackProvider(AgentProvider[None]):
    """Provider that processes messages through callbacks.

    Supports:
    - Sync and async callbacks
    - Optional context injection
    - String or ChatMessage returns
    """

    NAME = "callback"

    def __init__(
        self,
        callback: ProcessorCallback[Any],
        *,
        name: str = "",
        context: AgentContext[Any] | None = None,
        debug: bool = False,
    ):
        super().__init__(name=name or callback.__name__, debug=debug, context=context)
        self.callback = callback

    async def generate_response(
        self,
        *prompts: str | Content,
        message_history: list[ChatMessage],
        message_id: str,
        result_type: type[TResult] | None = None,
        system_prompt: str | None = None,
        usage_limits: UsageLimits | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Process message through callback."""
        text_prompts = [p for p in prompts if isinstance(p, str)]
        content_prompts = [p for p in prompts if isinstance(p, BaseContent)]

        # Get normal text prompt
        prompt = await format_prompts(text_prompts)

        try:
            # Create args tuple based on callback requirements
            args = (
                (self.context, prompt, *content_prompts)
                if has_argument_type(self.callback, AgentContext)
                else (prompt, *content_prompts)
            )
            raw = await execute(self.callback, *args, use_thread=True)
            # Handle potential awaitable result
            result = await raw if inspect.isawaitable(raw) else raw
            return ProviderResponse(content=result)

        except Exception as e:
            logger.exception("Processor callback failed")
            # Include callable name in error message
            name = getattr(self.callback, "__name__", str(self.callback))
            msg = f"Processor error in {name!r}: {e}"
            raise RuntimeError(msg) from e

    @asynccontextmanager
    async def stream_response(
        self,
        *prompts: str | Content,
        message_id: str,
        message_history: list[ChatMessage],
        result_type: type[Any] | None = None,
        system_prompt: str | None = None,
        usage_limits: UsageLimits | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingResponseProtocol]:
        """Simulate streaming by yielding complete result as one chunk."""

        class SingleChunkStream:
            def __init__(self, content: str):
                self.content = content
                self.is_complete = False
                self._streamed = False
                self.formatted_content = content
                self.model_name = "processor"

            def usage(self):
                return None

            async def stream(self):
                if not self._streamed:
                    self._streamed = True
                    yield self.content
                self.is_complete = True

            async def stream_text(self, delta: bool = False):
                if not self._streamed:
                    self._streamed = True
                    yield self.content
                self.is_complete = True

        try:
            # Get result using normal response generation
            result = await self.generate_response(
                *prompts, message_id=message_id, message_history=message_history
            )
            # we already save history etc with generate_response
            stream_result = SingleChunkStream(str(result.content))
            yield stream_result  # type: ignore

        except Exception as e:
            logger.exception("Processor streaming failed")
            msg = f"Processor error: {e}"
            raise RuntimeError(msg) from e


if __name__ == "__main__":
    # Example usage with streaming
    async def main():
        # Create processor
        from llmling_agent.agent.agent import Agent

        provider = CallbackProvider(str.upper, name="uppercase")
        uppercase = Agent[Any](provider=provider)

        # Normal usage
        result = await uppercase.run("hello")
        print(result.content)  # "HELLO"

        # Streaming usage
        async with uppercase.run_stream("hello") as stream:
            async for chunk in stream.stream():
                print(f"Chunk: {chunk}")  # Will print "HELLO" once

    asyncio.run(main())
