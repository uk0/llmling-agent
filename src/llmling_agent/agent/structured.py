"""LLMling integration with PydanticAI for AI-powered resource interaction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import TypeVar

from llmling_agent.log import get_logger
from llmling_agent.pydantic_ai_utils import to_result_schema


if TYPE_CHECKING:
    from pydantic_ai.agent import models
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.result import Usage

    from llmling_agent.agent.agent import Agent
    from llmling_agent.models.messages import ChatMessage


logger = get_logger(__name__)

TResult = TypeVar("TResult", default=str)
TDeps = TypeVar("TDeps", default=Any)


class StructuredAgent[TDeps, TResult]:
    """Wrapper for Agent that enforces a specific result type.

    This provides backwards compatibility for code that expects fixed result types,
    while keeping the base Agent class flexible.
    """

    def __init__(
        self,
        agent: Agent[TDeps],
        result_type: type[TResult],
        final_tool_name: str = "final_result",
        final_tool_description: str | None = None,
    ):
        self._agent = agent
        self._result_type = result_type
        self._tool_name = final_tool_name
        self._tool_description = final_tool_description

        # Set up initial schema
        schema = to_result_schema(
            result_type,
            final_tool_name,
            final_tool_description,
        )
        assert schema
        self._agent._pydantic_agent._result_schema = schema
        self._agent._pydantic_agent._allow_text_result = schema.allow_text_result

    async def run(
        self,
        *prompt: str,
        deps: TDeps | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        usage: Usage | None = None,
    ) -> ChatMessage[TResult]:
        """Run with fixed result type."""
        return await self._agent.run(
            *prompt,
            result_type=self._result_type,
            deps=deps,
            message_history=message_history,
            model=model,
            usage=usage,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)
