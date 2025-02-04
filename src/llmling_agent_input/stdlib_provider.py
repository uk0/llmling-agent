from __future__ import annotations

import json
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from llmling_agent_input.base import InputProvider


if TYPE_CHECKING:
    from llmling_agent.agent.context import AgentContext, ConfirmationResult
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.tools.base import ToolInfo


class StdlibInputProvider(InputProvider):
    """Input provider using only Python stdlib functionality."""

    async def get_input(
        self,
        context: AgentContext,
        prompt: str,
        result_type: type | None = None,
        message_history: list[ChatMessage] | None = None,
    ) -> str:
        if result_type:
            print(f"\nPlease provide response as {result_type.__name__}:")
        return input(f"{prompt}\n> ")

    async def get_tool_confirmation(
        self,
        context: AgentContext,
        tool: ToolInfo,
        args: dict[str, Any],
        message_history: list[ChatMessage] | None = None,
    ) -> ConfirmationResult:
        agent_name = context.node_name
        prompt = dedent(f"""
            Tool Execution Confirmation
            -------------------------
            Tool: {tool.name}
            Description: {tool.description or "No description"}
            Agent: {agent_name}

            Arguments:
            {json.dumps(args, indent=2)}

            Options:
            - y: allow execution
            - n/skip: skip this tool
            - abort: abort current run
            - quit: abort entire chain
            """).strip()

        response = input(f"{prompt}\nChoice [y/n/abort/quit]: ").lower()
        match response:
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
        msg = (
            "Multi-line code input not supported in stdlib provider. "
            "Use prompt-toolkit or textual provider instead."
        )
        raise NotImplementedError(msg)
