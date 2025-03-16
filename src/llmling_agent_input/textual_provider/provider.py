"""Textual input provider."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING, Any

from textual.app import App

from llmling_agent_input.base import InputProvider
from llmling_agent_input.textual_provider.code_input import CodeInputModal
from llmling_agent_input.textual_provider.confirmation_input import ConfirmationModal
from llmling_agent_input.textual_provider.run_input import InputModal


if TYPE_CHECKING:
    from textual.screen import ModalScreen

    from llmling_agent.agent.context import AgentContext, ConfirmationResult
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.tools.base import Tool


class UserCancelledError(Exception):
    """Raised when user cancels input."""


class BaseInputApp(App[str]):
    """Base app for standalone input."""

    def __init__(self, input_screen: ModalScreen[str]):
        super().__init__()
        self._input_screen = input_screen
        self._result: str | None = None

    async def on_mount(self):
        self._result = await self.push_screen_wait(self._input_screen)  # type: ignore
        self.exit()


class InputApp(BaseInputApp):
    """Standalone app for text input."""

    def __init__(self, prompt: str, result_type: type | None = None):
        super().__init__(InputModal(prompt, result_type))


class CodeInputApp(BaseInputApp):
    """Standalone app for code input."""

    def __init__(
        self,
        template: str | None = None,
        language: str = "python",
        description: str | None = None,
    ):
        super().__init__(CodeInputModal(template, language, description))


class ConfirmationApp(BaseInputApp):
    """Standalone app for confirmation."""

    def __init__(self, prompt: str):
        super().__init__(ConfirmationModal(prompt))


class TextualInputProvider(InputProvider):
    """Input provider using Textual modals or standalone app."""

    def __init__(
        self,
        app: App | None = None,
        real_streaming: bool = False,
    ):
        super().__init__(real_streaming=real_streaming)
        self.app = app

    async def get_input(
        self,
        context: AgentContext,
        prompt: str,
        result_type: type | None = None,
        message_history: list[ChatMessage] | None = None,
    ) -> Any:
        if self.app:
            result = await self.app.push_screen_wait(InputModal(prompt, result_type))
            if result is None:
                msg = "Input cancelled"
                raise UserCancelledError(msg)
            return result
        # Standalone mode - create temporary app
        app = InputApp(prompt, result_type)
        app_result = await app.run_async()
        if app_result is None:
            msg = "Input cancelled"
            raise UserCancelledError(msg)
        return app_result

    # async def _get_streaming_input(
    #     self,
    #     context: AgentContext,
    #     prompt: str,
    #     result_type: type | None = None,
    #     message_history: list[ChatMessage] | None = None,
    # ) -> AsyncIterator[str]:
    #     """Real-time streaming input using Textual."""
    #     from textual.app import App
    #     from textual.binding import Binding
    #     from textual.widgets import TextArea

    #     class StreamingInputModal(ModalScreen[str]):
    #         BINDINGS: ClassVar = [
    #             Binding("ctrl+enter", "submit", "Submit"),
    #             Binding("escape", "cancel", "Cancel"),
    #         ]

    #         def __init__(self, prompt: str, chunk_callback: Callable[[str], None]):
    #             super().__init__()
    #             self.prompt = prompt
    #             self._chunk_callback = chunk_callback

    #         def compose(self) -> ComposeResult:
    #             with Vertical(classes="modal-container"):
    #                 yield Label(self.prompt)
    #                 yield TextArea(id="input")

    #         def on_text_area_changed(self, event: TextArea.Changed):
    #             """Handle live updates."""
    #             self._chunk_callback(event.value)

    #         def action_submit(self):
    #             text = self.query_one(TextArea).text
    #             self.dismiss(text)

    #         def action_cancel(self):
    #             self.dismiss(None)

    #     chunk_queue: asyncio.Queue[str] = asyncio.Queue()

    #     async def handle_chunk(chunk: str):
    #         await chunk_queue.put(chunk)

    #     # Create modal or standalone app based on context
    #     if self.app:
    #         modal = StreamingInputModal(prompt, handle_chunk)
    #         content = await self.app.push_screen_wait(modal)
    #     else:

    #         class StreamingApp(App[str]):
    #             def __init__(self, prompt: str, callback: Callable[[str], None]):
    #                 super().__init__()
    #                 self._modal = StreamingInputModal(prompt, callback)
    #                 self._result: str | None = None

    #             async def on_mount(self):
    #                 self._result = await self.push_screen_wait(self._modal)
    #                 self.exit()

    #         app = StreamingApp(prompt, handle_chunk)
    #         content = await app.run_async()

    #     if content is None:
    #         msg = "Streaming input cancelled"
    #         raise UserCancelledError(msg)

    #     # Handle structured responses if needed
    #     if result_type and content:
    #         content = result_type.model_validate_json(content)  # type: ignore

    #     yield str(content)

    async def get_code_input(
        self,
        context: AgentContext,
        template: str | None = None,
        language: str = "python",
        description: str | None = None,
    ) -> str:
        if self.app:
            modal = CodeInputModal(template, language=language, description=description)
            result = await self.app.push_screen_wait(modal)
            if result is None:
                msg = "Code input cancelled"
                raise UserCancelledError(msg)
            return result
        app = CodeInputApp(template, language, description)
        app_result = await app.run_async()
        if app_result is None:
            msg = "Code input cancelled"
            raise UserCancelledError(msg)
        return app_result

    async def get_tool_confirmation(
        self,
        context: AgentContext,
        tool: Tool,
        args: dict[str, Any],
        message_history: list[ChatMessage] | None = None,
    ) -> ConfirmationResult:
        import anyenv

        prompt = dedent(f"""
            Tool Execution Confirmation
            -------------------------
            Tool: {tool.name}
            Description: {tool.description or "No description"}
            Agent: {context.node_name}

            Arguments:
            {anyenv.dump_json(args, indent=True)}
        """).strip()

        if self.app:
            result = await self.app.push_screen_wait(ConfirmationModal(prompt))
            return result or "skip"  # type: ignore
        app = ConfirmationApp(prompt)
        app_result = await app.run_async()
        return app_result or "skip"  # type: ignore
