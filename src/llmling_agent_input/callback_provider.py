"""Callback input provider implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import TYPE_CHECKING, Any

from llmling_agent.utils.inspection import execute
from llmling_agent_input.base import InputProvider


if TYPE_CHECKING:
    from llmling_agent.agent.context import AgentContext, ConfirmationResult
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.tools.base import Tool


class CallbackInputProvider(InputProvider):
    """Input provider that delegates to provided callbacks."""

    def __init__(
        self,
        get_input: Callable[..., str | Awaitable[str]] | None = None,
        get_streaming_input: Callable[..., AsyncIterator[str] | Iterator[str]]
        | None = None,
        get_tool_confirmation: Callable[
            ..., ConfirmationResult | Awaitable[ConfirmationResult]
        ]
        | None = None,
        get_code_input: Callable[..., str | Awaitable[str]] | None = None,
    ):
        self._get_input = get_input
        self._get_streaming = get_streaming_input
        self._get_confirmation = get_tool_confirmation
        self._get_code = get_code_input

    async def get_input(
        self,
        context: AgentContext,
        prompt: str,
        result_type: type | None = None,
        message_history: list[ChatMessage] | None = None,
    ) -> Any:
        if not self._get_input:
            return input(prompt)  # fallback
        return await execute(
            self._get_input,
            context=context,
            prompt=prompt,
            result_type=result_type,
            message_history=message_history,
        )

    async def _get_streaming_input(  # type: ignore
        self,
        context: AgentContext,
        prompt: str,
        result_type: type | None = None,
        message_history: list[ChatMessage] | None = None,
    ) -> AsyncIterator[str]:
        if not self._get_streaming:
            # Use parent class fallback
            return await super().get_streaming_input(  # type: ignore
                context, prompt, result_type, message_history
            )

        iterator = self._get_streaming(
            context=context,
            prompt=prompt,
            result_type=result_type,
            message_history=message_history,
        )

        if isinstance(iterator, AsyncIterator):
            return iterator

        async def wrap_sync():  # wrap sync iterator
            for item in iterator:
                yield item

        return wrap_sync()

    async def get_tool_confirmation(
        self,
        context: AgentContext,
        tool: Tool,
        args: dict[str, Any],
        message_history: list[ChatMessage] | None = None,
    ) -> ConfirmationResult:
        if not self._get_confirmation:
            return "allow"  # fallback: always allow
        return await execute(
            self._get_confirmation,
            context=context,
            tool=tool,
            args=args,
            message_history=message_history,
        )

    async def get_code_input(
        self,
        context: AgentContext,
        template: str | None = None,
        language: str = "python",
        description: str | None = None,
    ) -> str:
        if not self._get_code:
            return input("Enter code: ")  # basic fallback
        return await execute(
            self._get_code,
            context=context,
            template=template,
            language=language,
            description=description,
        )
