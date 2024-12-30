"""Test app for chat widgets."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.widgets import Header, Input

from llmling_agent.models import ChatMessage
from llmling_textual.widget import ChatView


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class ChatApp(App):
    """Test app with streaming."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield ChatView()
        yield Input(placeholder="Enter message...")

    async def simulate_stream(self, text: str) -> AsyncIterator[str]:
        """Simulate streaming response."""
        for word in text.split():
            yield word + " "
            await asyncio.sleep(0.2)  # Simulate network delay

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        chat = self.query_one(ChatView)
        # Add user message
        await chat.add_message(ChatMessage(role="user", content=event.value, name="User"))

        # Create assistant message with empty content
        msg = ChatMessage[str](
            role="assistant",
            content="",  # Start empty
            name="Assistant",
            model="gpt-4",
        )

        # Create the message widget first
        widget = await chat.add_message(msg)

        # Stream the response
        response = "Let me think about that.. Here's a thoughtful response, word by word."
        content = ""
        async for chunk in self.simulate_stream(response):
            content += chunk  # Accumulate content
            widget.content = content  # Update widget with full content
            widget.scroll_visible()

        event.input.value = ""


if __name__ == "__main__":
    app = ChatApp()
    app.run()
