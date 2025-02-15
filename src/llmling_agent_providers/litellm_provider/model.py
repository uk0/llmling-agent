"""LiteLLM model wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from pydantic import BaseModel


if TYPE_CHECKING:
    from litellm import ModelResponse
    from litellm.utils import CustomStreamWrapper
    from pydantic import BaseModel as PydanticModel


class LiteLLMKwargs(TypedDict):
    """Common kwargs for LiteLLM completions."""

    max_tokens: NotRequired[int | None]
    temperature: NotRequired[float | None]
    response_format: NotRequired[type[PydanticModel] | None]
    tools: NotRequired[list[dict[str, Any]] | None]
    tool_choice: NotRequired[str | None]
    num_retries: NotRequired[int]


class LiteLLMModel(BaseModel):
    """LiteLLM model wrapper for consistent model behavior."""

    def __init__(self, identifier: str):
        self._identifier = identifier.replace(":", "/")  # litellm format

    @property
    def model_name(self) -> str:
        return self._identifier

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: type[PydanticModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        num_retries: int = 1,
        **extra_kwargs: Any,
    ) -> ModelResponse:
        """Complete using litellm.

        Args:
            messages: List of message dicts (role, content)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            response_format: Optional Pydantic model for response
            tools: List of tool definitions
            tool_choice: Tool choice strategy ("auto" or "none")
            num_retries: Number of retries on failure
            **extra_kwargs: Additional kwargs passed to litellm
        """
        from litellm import acompletion

        return await acompletion(  # type: ignore
            model=self.model_name,
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice if tools else None,
            num_retries=num_retries,
            **extra_kwargs,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: type[PydanticModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        num_retries: int = 1,
        **extra_kwargs: Any,
    ) -> CustomStreamWrapper:
        """Stream completions using litellm.

        Args:
            messages: List of message dicts (role, content)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            response_format: Optional Pydantic model for response
            tools: List of tool definitions
            tool_choice: Tool choice strategy ("auto" or "none")
            num_retries: Number of retries on failure
            **extra_kwargs: Additional kwargs passed to litellm
        """
        from litellm import acompletion

        return await acompletion(  # pyright: ignore
            model=self.model_name,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice if tools else None,
            num_retries=num_retries,
            **extra_kwargs,
        )
