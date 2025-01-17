from __future__ import annotations

from typing import TYPE_CHECKING

from slashed import OutputWriter
from textual.app import App
from textual.widgets import Input, RichLog

from llmling_agent.delegation.commands import ListAgentsCommand
from llmling_agent.delegation.supervisor import InputHandler, PoolSupervisor
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.delegation.pool import AgentPool


logger = get_logger(__name__)


class TextualInputHandler(InputHandler):
    """Input handler using Textual Input widget."""

    def __init__(self, app: SupervisorApp):
        self.app = app

    async def get_input(self, prompt: str) -> str:
        """Get input from Textual widget."""
        return self.app.input.value


class TextualOutputHandler(OutputWriter):
    """Output handler using Textual RichLog."""

    def __init__(self, app: SupervisorApp):
        self.app = app

    async def print(self, message: str):
        """Display message in RichLog."""
        self.app.call_later(self.app.output.write, message)


class SupervisorApp(App):
    """Basic supervision interface."""

    CSS = """
    RichLog { height: 1fr; }
    Input { dock: bottom; }
    """

    def __init__(self, pool: AgentPool):
        super().__init__()
        self.pool = pool
        self.supervisor: PoolSupervisor | None = None

    def compose(self):
        """Create and yield widgets."""
        self.output = RichLog()
        self.input = Input(placeholder="@agent message or /command")
        yield self.output
        yield self.input

    async def on_mount(self):
        """Start supervision when app mounts."""
        # Create supervisor with proper handlers
        self.supervisor = PoolSupervisor(
            pool=self.pool,
            input_handler=TextualInputHandler(self),
            output_handler=TextualOutputHandler(self),
        )
        # Register commands
        self.supervisor.commands.register_command(ListAgentsCommand())
        await self.supervisor.start()

    async def on_input_submitted(self, event: Input.Submitted):
        """Process submitted input."""
        if not self.supervisor:
            return

        text = event.value.strip()
        if not text:
            return

        if text.lower() == "exit":
            self.exit()
        else:
            await self.supervisor._handle_command(text)
        self.input.value = ""

    async def on_unmount(self):
        """Clean up when app exits."""
        if self.supervisor:
            await self.supervisor.stop()


if __name__ == "__main__":
    import asyncio
    import sys

    from llmling_agent.delegation.pool import AgentPool

    model = "openai:gpt-4o-mini"

    async def main():
        async with AgentPool[None]() as pool:
            prompt = "You are a pirate."
            analyzer = await pool.add_agent("analyzer", model=model, system_prompt=prompt)
            prompt = "You are a hippie."
            _planner = await pool.add_agent("planner", model=model, system_prompt=prompt)
            await analyzer.run_continuous("Tell a joke", max_count=5, interval=10)
            pool.start_supervision()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSupervision ended.")
    except Exception:
        logger.exception("Supervision failed")
        sys.exit(1)
