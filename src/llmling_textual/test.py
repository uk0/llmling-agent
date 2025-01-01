from __future__ import annotations

import asyncio
import logging
from typing import Any

import logfire
from slashed.textual_adapter import CommandInput
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header

from llmling_agent import LLMlingAgent
from llmling_agent.commands import get_commands
from llmling_agent.models import ChatMessage
from llmling_textual.widget import ChatView


logger = logging.getLogger(__name__)


class ChatApp(App):
    """Chat application with command support."""

    def __init__(self):
        super().__init__()
        self._pending_tasks: set[asyncio.Task[None]] = set()
        self._agent: LLMlingAgent[Any, str] | None = None
        self._agent_cm: Any = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(ChatView(), id="chat-output")
        yield CommandInput(
            output_id="chat-output",
            context_data=self._agent,
            status_id="status-area",
        )

    async def on_mount(self):
        """Initialize agent when app starts."""
        path = "src/llmling_agent/config_resources/agents.yml"
        self._agent_cm = LLMlingAgent.open_agent(path, "url_opener")
        self._agent = await self._agent_cm.__aenter__()
        assert self._agent
        logger.info("Agent initialized: %s", self._agent.name)
        command_input = self.query_one(CommandInput)
        for command in get_commands():
            command_input.store.register_command(command)

    async def on_unmount(self):
        """Clean up tasks and agent."""
        for task in self._pending_tasks:
            task.cancel()
        if self._pending_tasks:
            await asyncio.wait(self._pending_tasks)

        if self._agent_cm is not None:
            await self._agent_cm.__aexit__(None, None, None)

    def _create_task(self, coro: Any):
        """Create and track a task."""
        task = asyncio.create_task(coro)
        logger.debug("Created task: %s", task.get_name())

        def _done_callback(t: asyncio.Task[None]):
            logger.debug("Task completed: %s", t.get_name())
            self._pending_tasks.discard(t)
            if t.exception():
                logger.error("Task failed with error: %s", t.exception())

        task.add_done_callback(_done_callback)
        self._pending_tasks.add(task)

    @logfire.instrument("Got input: {event.text}")
    @on(CommandInput.InputSubmitted)
    async def handle_submit(self, event: CommandInput.InputSubmitted):
        """Handle regular input submission."""
        self._create_task(self.handle_chat_message(event.text))

    @logfire.instrument("Processing message: {text}")
    async def handle_chat_message(self, text: str):
        """Process normal chat messages."""
        if not self._agent:
            logger.error("No agent available!")
            return
        chat = self.query_one(ChatView)

        # Add user message
        await chat.add_message(ChatMessage[str](role="user", content=text, name="User"))

        # Create empty assistant message, will get populated for streaming.
        name = self._agent.model_name
        msg = ChatMessage[str](role="assistant", content="", name="Assistant", model=name)
        widget = await chat.add_message(msg)

        try:
            # Stream the response from the agent
            logger.debug("Starting stream")
            async with self._agent.run_stream(text) as stream:
                full_content = ""
                logger.debug("Stream opened, waiting for chunks")
                async for chunk in stream.stream():
                    # Get just the new content by removing the overlap
                    new_content = str(chunk).removeprefix(full_content)
                    logger.debug("Got new content: %r", new_content)

                    full_content += new_content
                    widget.update_content(full_content)
                    widget.refresh()

                logger.debug("Stream completed with final content: %r", full_content)
        except Exception:
            logger.exception("Error streaming response")
            widget.update_content("Error processing response")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        filename="chat_app.log",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    app = ChatApp()
    app.run()
