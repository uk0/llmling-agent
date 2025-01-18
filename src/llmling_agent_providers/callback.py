from __future__ import annotations

from collections.abc import Awaitable
from contextlib import asynccontextmanager
import inspect
from typing import TYPE_CHECKING, Any, TypeVar

from llmling_agent.log import get_logger
from llmling_agent.utils.inspection import has_argument_type
from llmling_agent_providers.base import AgentProvider, ProviderResponse


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.result import StreamedRunResult

    from llmling_agent.models.content import Content
    from llmling_agent.models.providers import ProcessorCallback


logger = get_logger(__name__)

TResult = TypeVar("TResult")
TDeps = TypeVar("TDeps")


class CallbackProvider[TDeps](AgentProvider[TDeps]):
    """Provider that processes messages through callbacks.

    Supports:
    - Sync and async callbacks
    - Optional context injection
    - String or ChatMessage returns
    """

    def __init__(
        self,
        callback: ProcessorCallback[Any],
        *,
        name: str = "processor",
        debug: bool = False,
    ):
        super().__init__(name=name, debug=debug)
        self.callback = callback
        self._wants_context = has_argument_type(callback, "AgentContext")
        self._is_async = inspect.iscoroutinefunction(callback)

    async def generate_response(
        self,
        *prompts: str | Content,
        message_id: str,
        result_type: type[TResult] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Process message through callback."""
        try:
            # Create args tuple based on callback requirements
            args = (self.context, *prompts) if self._wants_context else (*prompts,)
            raw_result = self.callback(*args)

            # Handle async/sync result
            result = await raw_result if isinstance(raw_result, Awaitable) else raw_result

            return ProviderResponse(content=result)

        except Exception as e:
            logger.exception("Processor callback failed")
            msg = f"Processor error: {e}"
            raise RuntimeError(msg) from e

    @asynccontextmanager
    async def stream_response(
        self,
        *prompts: str | Content,
        message_id: str,
        result_type: type[Any] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamedRunResult]:
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

        try:
            # Get result using normal response generation
            result = await self.generate_response(*prompts, message_id=message_id)
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

        provider = CallbackProvider[Any](str.upper, name="uppercase")
        uppercase = Agent[Any](provider=provider)

        # Normal usage
        result = await uppercase.run("hello")
        print(result.content)  # "HELLO"

        # Streaming usage
        async with uppercase.run_stream("hello") as stream:
            async for chunk in stream.stream():
                print(f"Chunk: {chunk}")  # Will print "HELLO" once

    import asyncio

    asyncio.run(main())
