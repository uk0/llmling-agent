from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from slashed.textual_adapter import CommandInput
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header

from llmling_agent.models import ChatMessage
from llmling_textual.widget import ChatView


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class ChatApp(App):
    """Chat application with command support."""

    def __init__(self) -> None:
        super().__init__()
        self._pending_tasks: set[asyncio.Task[None]] = set()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(ChatView(), id="chat-output")
        yield CommandInput(
            output_id="chat-output",
            status_id="status-area",
        )

    def _create_task(self, coro: Any) -> None:
        """Create and track a task."""
        task = asyncio.create_task(coro)

        def _done_callback(t: asyncio.Task[None]) -> None:
            self._pending_tasks.discard(t)

        task.add_done_callback(_done_callback)
        self._pending_tasks.add(task)

    async def on_command_input_input_submitted(
        self, message: CommandInput.InputSubmitted
    ) -> None:
        """Handle non-command input."""
        self._create_task(self.handle_chat_message(message.text))

    async def handle_chat_message(self, text: str) -> None:
        """Process normal chat messages."""
        chat = self.query_one(ChatView)

        # Add user message
        msg = ChatMessage[str](role="user", content=text, name="User")
        await chat.add_message(msg)

        # Create assistant message and stream response
        msg = ChatMessage[str](
            role="assistant",
            content="",
            name="Assistant",
            model="gpt-4",
        )
        widget = await chat.add_message(msg)

        # Simulate response
        content = ""
        async for chunk in self.simulate_stream(
            "This is a regular chat response (not a command)..."
        ):
            content += chunk
            widget.content = content
            widget.scroll_visible()

    async def simulate_stream(self, text: str) -> AsyncIterator[str]:
        """Simulate streaming response."""
        for word in text.split():
            yield word + " "
            await asyncio.sleep(0.2)

    async def on_unmount(self) -> None:
        """Clean up pending tasks when app exits."""
        for task in self._pending_tasks:
            task.cancel()
        if self._pending_tasks:
            # Wait for tasks to complete
            await asyncio.wait(self._pending_tasks)


if __name__ == "__main__":
    app = ChatApp()
    app.run()
