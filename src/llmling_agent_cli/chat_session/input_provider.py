"""CLI session input provider."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING, Any

from llmling_agent_input.base import InputProvider


if TYPE_CHECKING:
    from llmling_agent.agent.context import AgentContext, ConfirmationResult
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.tools.base import Tool
    from llmling_agent_cli.chat_session.session import InteractiveSession


class CLISessionInputProvider(InputProvider):
    """Input provider integrated with our CLI session."""

    def __init__(self, session: InteractiveSession):
        self.session = session

    async def get_input(
        self,
        context: AgentContext,
        prompt: str,
        result_type: type | None = None,
        message_history: list[ChatMessage] | None = None,
    ) -> Any:
        # Use session's prompt and formatting
        if result_type:
            self.session.console.print(
                f"\nPlease provide response as {result_type.__name__}:"
            )

        assert self.session._prompt  # should be set up by InteractiveSession
        return await self.session._prompt.prompt_async(prompt + "\n> ")

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

            Options:
            - y: allow execution
            - n/skip: skip this tool
            - abort: abort current run
            - quit: abort entire chain
            """).strip()

        self.session.console.print(prompt)
        assert self.session._prompt
        response = await self.session._prompt.prompt_async("Choice [y/n/abort/quit]: ")

        match response.lower():
            case "y" | "yes":
                return "allow"
            case "abort":
                return "abort_run"
            case "quit":
                return "abort_chain"
            case _:
                return "skip"

    async def get_code_input(
        self,
        context: AgentContext,
        template: str | None = None,
        language: str = "python",
        description: str | None = None,
    ) -> str:
        # We might want to use a multiline session here
        from prompt_toolkit import PromptSession
        from prompt_toolkit.lexers import PygmentsLexer
        from pygments.lexers.python import PythonLexer

        if description:
            self.session.console.print(f"\n{description}")
        lexer = PygmentsLexer(PythonLexer)
        session = PromptSession[Any](multiline=True, lexer=lexer)

        return await session.prompt_async(
            "\nEnter code (ESC + Enter to submit):\n",
            default=template or "",
        )
