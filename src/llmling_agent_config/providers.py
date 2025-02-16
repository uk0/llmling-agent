"""Provider configuration models."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ImportString

from llmling_agent.common_types import EndStrategy, ModelProtocol  # noqa: TC001
from llmling_agent_models import AnyModelConfig  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent_providers.base import AgentProvider
    from llmling_agent_providers.callback import CallbackProvider

type ProcessorCallback[TResult] = Callable[..., TResult | Awaitable[TResult]]


class BaseProviderConfig(BaseModel):
    """Base configuration for agent providers.

    Common settings that apply to all provider types, regardless of their
    specific implementation. Provides basic identification and configuration
    options that every provider should have.
    """

    type: str = Field(init=False)
    """Type discriminator for provider configs."""

    name: str | None = None
    """Optional name for the provider instance."""

    model_settings: ModelSettings | None = None
    """Optional settings to configure the LLM behavior."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")


class PydanticAIProviderConfig(BaseProviderConfig):
    """Configuration for PydanticAI-based provider.

    This provider uses PydanticAI for handling model interactions, tool calls,
    and structured outputs. It provides fine-grained control over model behavior
    and validation.
    """

    type: Literal["pydantic_ai"] = Field("pydantic_ai", init=False)
    """Pydantic-AI provider."""

    end_strategy: EndStrategy = "early"
    """How to handle tool calls when final result found:
    - early: Stop when valid result found
    - complete: Run all requested tools
    - confirm: Ask user what to do
    """

    model: str | AnyModelConfig | None = None
    """Optional model name to use. If not specified, uses default model."""

    result_retries: int | None = None
    """Maximum retries for result validation.
    None means use the global retry setting.
    """

    defer_model_check: bool = False
    """Whether to defer model evaluation until first run.
    True can speed up initialization but might fail later.
    """

    validation_enabled: bool = True
    """Whether to validate model outputs against schemas."""

    allow_text_fallback: bool = True
    """Whether to accept plain text when structured output fails."""

    def get_provider(self) -> AgentProvider:
        """Create PydanticAI provider instance."""
        from llmling_agent_models.base import BaseModelConfig
        from llmling_agent_providers.pydanticai import PydanticAIProvider

        settings = (
            self.model_settings.model_dump(exclude_none=True)
            if self.model_settings
            else {}
        )
        match self.model:
            case str():
                model: str | ModelProtocol | None = self.model
            case BaseModelConfig():
                model = self.model.get_model()
            case _:
                model = None
        return PydanticAIProvider(
            model=model,
            name=self.name or "ai-agent",
            end_strategy=self.end_strategy,
            result_retries=self.result_retries,
            defer_model_check=self.defer_model_check,
            model_settings=settings,
        )


class LiteLLMProviderConfig(BaseProviderConfig):
    """Configuration for LiteLLM-based provider."""

    type: Literal["litellm"] = Field("litellm", init=False)
    """LiteLLM provider."""

    retries: int = 1
    """Maximum retries for model calls."""

    model: str | None = None
    """Optional model name to use. If not specified, uses default model."""

    def get_provider(self) -> AgentProvider:
        """Create PydanticAI provider instance."""
        from llmling_agent_providers.litellm_provider import LiteLLMProvider

        settings = {}
        if self.model_settings:
            settings = {
                "max_tokens": self.model_settings.max_tokens,
                "temperature": self.model_settings.temperature,
                "top_p": self.model_settings.top_p,
                "request_timeout": self.model_settings.timeout,  # different name!
                "presence_penalty": self.model_settings.presence_penalty,
                "frequency_penalty": self.model_settings.frequency_penalty,
                "seed": self.model_settings.seed,
            }
            # Remove None values
            settings = {k: v for k, v in settings.items() if v is not None}

        name = self.name or "ai-agent"
        return LiteLLMProvider(
            name=name,
            model=self.model,
            retries=self.retries,
            model_settings=settings,
        )


class HumanProviderConfig(BaseProviderConfig):
    """Configuration for human-in-the-loop provider.

    This provider enables direct human interaction for responses and decisions.
    Useful for testing, training, and oversight of agent operations.
    """

    type: Literal["human"] = Field("human", init=False)
    """Human-input provider."""

    timeout: int | None = None
    """Timeout in seconds for human response. None means wait indefinitely."""

    show_context: bool = True
    """Whether to show conversation context to human."""

    def get_provider(self) -> AgentProvider:
        """Create human provider instance."""
        from llmling_agent_providers.human import HumanProvider

        return HumanProvider(
            name=self.name or "human-agent",
            timeout=self.timeout,
            show_context=self.show_context,
        )


class CallbackProviderConfig[TResult](BaseProviderConfig):
    """Configuration for callback-based provider.

    Allows defining processor functions through:
    - Import path to callback function
    - Generic type for result validation
    """

    type: Literal["callback"] = Field("callback", init=False)
    """Import-path based Callback provider."""

    callback: ImportString[ProcessorCallback[TResult]]
    """Import path to processor callback."""

    def get_provider(self) -> CallbackProvider:
        """Create callback provider instance."""
        from llmling_agent_providers.callback import CallbackProvider

        name = self.name or self.callback.__name__
        return CallbackProvider(self.callback, name=name)


# The union type used in AgentConfig
ProviderConfig = Annotated[
    PydanticAIProviderConfig
    | HumanProviderConfig
    | LiteLLMProviderConfig
    | CallbackProviderConfig,
    Field(discriminator="type"),
]


class ModelSettings(BaseModel):
    """Settings to configure an LLM."""

    max_tokens: int | None = None
    """The maximum number of tokens to generate."""

    temperature: float | None = Field(None, ge=0.0, le=2.0)
    """Amount of randomness in the response (0.0 - 2.0)."""

    top_p: float | None = Field(None, ge=0.0, le=1.0)
    """An alternative to sampling with temperature, called nucleus sampling."""

    timeout: float | None = None
    """Override the client-level default timeout for a request, in seconds."""

    parallel_tool_calls: bool | None = None
    """Whether to allow parallel tool calls."""

    seed: int | None = None
    """The random seed to use for the model."""

    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    """Penalize new tokens based on whether they have appeared in the text so far."""

    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    """Penalize new tokens based on their existing frequency in the text so far."""

    logit_bias: dict[str, int] | None = None
    """Modify the likelihood of specified tokens appearing in the completion."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to TypedDict format for pydantic-ai."""
        return {k: v for k, v in self.model_dump().items() if v is not None}
