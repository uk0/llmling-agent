"""Agent provider implementations."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from llmling import ToolError
import logfire

from llmling_agent.log import get_logger
from llmling_agent_providers import AgentProvider
from llmling_agent_providers.base import ProviderResponse


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.result import StreamedRunResult

    from llmling_agent.agent.conversation import ConversationManager
    from llmling_agent.common_types import ModelType
    from llmling_agent.models.context import AgentContext
    from llmling_agent.tools.manager import ToolManager


logger = get_logger(__name__)


def get_textual_streaming_app():
    from textual.app import App
    from textual.events import Key  # noqa: TC002
    from textual.widgets import Input

    class StreamingInputApp(App):
        def __init__(self, chunk_callback):
            super().__init__()
            self.chunk_callback = chunk_callback
            self.buffer = []
            self.done = False

        def compose(self):
            yield Input(id="input")

        def on_input_changed(self, event: Input.Changed):
            # New character was typed
            if len(event.value) > len(self.buffer):
                new_char = event.value[len(self.buffer) :]
                self.chunk_callback(new_char)
            self.buffer = list(event.value)

        def on_key(self, event: Key):
            if event.key == "enter":
                self.done = True
                self.result = "".join(self.buffer)
                self.exit()

    return StreamingInputApp


class HumanProvider(AgentProvider):
    """Provider for human-in-the-loop responses."""

    model = None
    _conversation: ConversationManager

    def __init__(
        self,
        *,
        conversation: ConversationManager,
        context: AgentContext[Any],
        tools: ToolManager,
        name: str = "human",
        timeout: int | None = None,
        show_context: bool = True,
        debug: bool = False,
    ):
        """Initialize human provider."""
        super().__init__(
            tools=tools,
            conversation=conversation,
            model=None,
            context=context,
        )
        self.name = name or "human"
        self._debug = debug
        self._timeout = timeout
        self._show_context = show_context

    def __repr__(self) -> str:
        return f"Human({self.name!r})"

    @logfire.instrument("Human input. result type {result_type}. Prompt: {prompt}")
    async def generate_response(
        self,
        prompt: str,
        message_id: str,
        *,
        result_type: type[Any] | None = None,
        model: ModelType = None,
    ) -> ProviderResponse:
        """Get response through human input.

        Args:
            prompt: Text prompt to respond to
            message_id: Message id to use for the response
            result_type: Optional type for structured responses
            model: Model override (unused for human)
        """
        if self._show_context:
            history = await self._conversation.format_history(
                format_template="[{agent}] {content}\n",  # Optionally customize format
                include_system=False,  # Skip system messages for cleaner output
            )
            if history:
                print("\nContext:")
                print(history)
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
                error_msg = f"Invalid response format: {e}"
                raise ToolError(error_msg) from e

        return ProviderResponse(content=content, tool_calls=[], usage=None)

    @asynccontextmanager
    async def stream_response(
        self,
        prompt: str,
        message_id: str,
        *,
        result_type: type[Any] | None = None,
        model: ModelType = None,
    ) -> AsyncIterator[StreamedRunResult]:  # type: ignore[type-var]
        """Stream response keystroke by keystroke."""
        print(f"\n{prompt}")
        if result_type:
            print(f"(Please provide response as {result_type.__name__})")

        # Create a StreamedRunResult-like object
        class StreamResult:
            def __init__(self):
                self.stream = None
                self.is_complete = False
                self.formatted_content = ""
                self.is_structured = False
                self.model_name = "human"

            def usage(self):
                return None

        stream_result = StreamResult()
        chunk_queue: asyncio.Queue[str] = asyncio.Queue()

        async def handle_chunk(chunk: str):
            await chunk_queue.put(chunk)

        # Setup streaming
        async def wrapped_stream(*args, **kwargs):
            while not stream_result.is_complete or not chunk_queue.empty():
                try:
                    chunk = await chunk_queue.get()
                    self.chunk_streamed.emit(chunk, message_id)
                    yield chunk
                except asyncio.CancelledError:
                    break

        stream_result.stream = wrapped_stream  # type: ignore

        try:
            # Run textual app
            textual_app = get_textual_streaming_app()
            app = textual_app(handle_chunk)
            content = await app.run_async()

            # Mark as complete and set final content
            stream_result.is_complete = True

            # Parse structured response if needed
            if result_type:
                try:
                    content = result_type.model_validate_json(content)
                    stream_result.is_structured = True
                except Exception as e:
                    logger.exception("Failed to parse structured response")
                    error_msg = f"Invalid response format: {e}"
                    raise ToolError(error_msg) from e

            stream_result.formatted_content = str(content)
            yield stream_result  # type: ignore

        finally:
            # Cleanup if needed
            stream_result.is_complete = True
