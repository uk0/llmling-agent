"""LLMling integration with PydanticAI for AI-powered resource interaction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import TypeVar

from llmling_agent.log import get_logger
from llmling_agent.model_utils import format_instance_for_llm
from llmling_agent.responses.models import (
    BaseResponseDefinition,
    ResponseDefinition,
)
from llmling_agent.responses.utils import to_type


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

    This wrapper ensures the agent always returns results of the specified type.
    The type can be provided as:
    - A Python type for validation
    - A response definition name from the manifest
    - A complete response definition instance
    """

    def __init__(
        self,
        agent: Agent[TDeps],
        result_type: type[TResult] | str | ResponseDefinition,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ):
        """Initialize structured agent wrapper.

        Args:
            agent: Base agent to wrap
            result_type: Expected result type:
                - BaseModel / dataclasses
                - Name of response definition in manifest
                - Complete response definition instance
            tool_name: Optional override for tool name
            tool_description: Optional override for tool description

        Raises:
            ValueError: If named response type not found in manifest
        """
        logger.debug("StructuredAgent.run result_type = %s", result_type)
        self._agent = agent
        self._result_type = to_type(result_type)
        agent.set_result_type(result_type)

        match result_type:
            case type() | str():
                # For types and named definitions, use overrides if provided
                self._agent.set_result_type(
                    result_type,
                    tool_name=tool_name,
                    tool_description=tool_description,
                )
            case BaseResponseDefinition():
                # For response definitions, use as-is
                # (overrides don't apply to complete definitions)
                self._agent.set_result_type(result_type)

    async def run(
        self,
        *prompt: str | TResult,
        result_type: type[TResult] | None = None,
        deps: TDeps | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        usage: Usage | None = None,
    ) -> ChatMessage[TResult]:
        """Run with fixed result type.

        Args:
            prompt: String prompts or structured objects of type TResult
            result_type: Expected result type:
                - BaseModel / dataclasses
                - Name of response definition in manifest
                - Complete response definition instance
            deps: Optional dependencies for the agent
            message_history: Optional previous messages for context
            model: Optional model override
            usage: Optional usage tracking
        """
        formatted_prompts = [
            format_instance_for_llm(p) if not isinstance(p, str) else p for p in prompt
        ]

        return await self._agent.run(
            *formatted_prompts,
            result_type=result_type or self._result_type,
            deps=deps,
            message_history=message_history,
            model=model,
            usage=usage,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)
