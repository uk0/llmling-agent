"""Agent provider implementations."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from llmling import ToolError
from slashed import CommandStore, DefaultOutputWriter, parse_command

from llmling_agent.log import get_logger
from llmling_agent.observability import track_action
from llmling_agent.prompts.convert import format_prompts
from llmling_agent_providers.base import (
    AgentProvider,
    ProviderResponse,
    StreamingResponseProtocol,
    UsageLimits,
)
from llmling_agent_providers.human.utils import get_textual_streaming_app


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling_agent.agent.context import AgentContext
    from llmling_agent.common_types import ModelType
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.models.content import Content


logger = get_logger(__name__)


class HumanProvider(AgentProvider):
    """Provider for human-in-the-loop responses."""

    model = None
    NAME = "human"

    def __init__(
        self,
        *,
        name: str = "human",
        context: AgentContext | None = None,
        timeout: int | None = None,
        show_context: bool = True,
        command_store: CommandStore | None = None,
        use_promptantic: bool = True,
        debug: bool = False,
    ):
        """Initialize human provider."""
        from llmling_agent_commands import get_commands

        super().__init__(context=context)
        self.name = name or "human"
        self._debug = debug
        self._timeout = timeout
        self._show_context = show_context
        self.use_promptantic = use_promptantic
        self.commands = command_store or CommandStore()
        for cmd in get_commands():
            self.commands.register_command(cmd)

    def __repr__(self) -> str:
        return f"Human({self.name!r})"

    @track_action("Human input. result type {result_type}")
    async def generate_response(
        self,
        *prompts: str | Content,
        message_id: str,
        message_history: list[ChatMessage],
        result_type: type[Any] | None = None,
        model: ModelType = None,
        system_prompt: str | None = None,
        usage_limits: UsageLimits | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Get response through human input.

        Args:
            prompts: Text / Image Content to respond to
            message_id: Message id to use for the response
            message_history: Message history
            result_type: Optional type for structured responses
            model: Model override (unused for human)
            system_prompt: System prompt from SystemPrompts manager
            usage_limits: Usage limits for the model
            kwargs: Additional arguments for human (unused)
        """
        # Show prompt and get response
        formatted = await format_prompts(prompts)
        content = await self.context.get_input_provider().get_input(
            self.context,
            formatted,
            result_type=result_type,
            message_history=message_history,
        )
        return ProviderResponse(content=content)

    @asynccontextmanager
    async def stream_response(
        self,
        *prompts: str | Content,
        message_id: str,
        message_history: list[ChatMessage],
        result_type: type[Any] | None = None,
        model: ModelType = None,
        system_prompt: str | None = None,
        usage_limits: UsageLimits | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingResponseProtocol]:
        """Stream response keystroke by keystroke."""
        prompt = await format_prompts(prompts)
        print(f"\n{prompt}")
        if result_type:
            print(f"(Please provide response as {result_type.__name__})")

        # Create a StreamingResponseProtocol-like object
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

    async def handle_input(self, content: str):
        """Handle all human input."""
        from llmling_agent.messaging.events import UIEventData

        if not content.strip():
            return

        # Store for generate_response if needed
        self._last_response = content

        try:
            if content.startswith("/"):
                # Regular command
                parsed = parse_command(content[1:])
                _event = UIEventData(
                    source=self.name,
                    type="command",
                    content=parsed.name,
                    args=parsed.args.args,
                    kwargs=parsed.args.kwargs,
                )
                await self.commands.execute_command_with_context(
                    parsed.name,
                    context=self.context,
                    output_writer=DefaultOutputWriter(),
                )

            elif content.startswith("@"):
                # Agent-specific interaction
                agent_name, message = content[1:].split(maxsplit=1)
                if not self.context.pool:
                    logger.error("No agent pool available")
                    return

                agent = self.context.pool.get_agent(agent_name)
                if message.startswith("/"):
                    # Command for specific agent
                    parsed = parse_command(message[1:])
                    _event = UIEventData(
                        source=self.name,
                        type="agent_command",
                        content=parsed.name,
                        args=parsed.args.args,
                        kwargs=parsed.args.kwargs,
                        agent_name=agent_name,
                    )
                    await self.commands.execute_command_with_context(
                        parsed.name,
                        context=agent.context,
                        output_writer=DefaultOutputWriter(),
                    )
                else:
                    # Message for specific agent
                    _event = UIEventData(
                        source=self.name,
                        type="agent_message",
                        content=message,
                        agent_name=agent_name,
                    )
                    await agent.run(message)

            else:
                # Regular message
                _event = UIEventData(source=self.name, type="message", content=content)

            # self.agent.message_received.emit(event)

        except Exception:
            logger.exception("Failed to handle input")
