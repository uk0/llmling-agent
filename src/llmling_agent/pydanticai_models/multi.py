"""Multi-model implementations."""

from __future__ import annotations

from contextlib import asynccontextmanager
import random
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, model_validator
from pydantic_ai.models import AgentModel, KnownModelName, Model, infer_model

from llmling_agent.log import get_logger
from llmling_agent.pydanticai_models.base import PydanticModel


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.messages import Message, ModelAnyResponse
    from pydantic_ai.models import EitherStreamedResponse
    from pydantic_ai.tools import ToolDefinition


logger = get_logger(__name__)


class MultiModel(PydanticModel):
    """Base for model configurations that combine multiple language models."""

    type: str = Field(description="Discriminator field for multi-model types")
    models: list[KnownModelName] = Field(
        description="List of models to use",
        min_length=1,  # Require at least one model
    )

    def name(self) -> str:
        """Get model name."""
        return f"multi-{self.type}"

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create agent model implementation."""
        raise NotImplementedError

    @property
    def available_models(self) -> list[Model]:
        """Get list of available models."""
        return [infer_model(name) for name in self.models]  # type: ignore[arg-type]


class MultiAgentModel(AgentModel):
    """AgentModel implementation for multi-model setups."""

    def __init__(
        self,
        models: list[Model],
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> None:
        """Initialize multi-agent model."""
        if not models:
            msg = "At least one model must be provided"
            raise ValueError(msg)
        self.models = models
        self.function_tools = function_tools
        self.allow_text_result = allow_text_result
        self.result_tools = result_tools
        self._initialized_models: list[AgentModel] | None = None

    async def _initialize_models(self) -> list[AgentModel]:
        """Initialize all agent models."""
        if self._initialized_models is None:
            self._initialized_models = []
            for model in self.models:
                agent_model = await model.agent_model(
                    function_tools=self.function_tools,
                    allow_text_result=self.allow_text_result,
                    result_tools=self.result_tools,
                )
                self._initialized_models.append(agent_model)
        return self._initialized_models


class RandomAgentModel(MultiAgentModel):
    """Randomly selects from available models."""

    async def request(
        self,
        messages: list[Message],
    ) -> tuple[ModelAnyResponse, Any]:
        """Make request using randomly selected model."""
        models = await self._initialize_models()
        selected = random.choice(models)
        logger.debug("Selected model: %s", selected)
        return await selected.request(messages)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[Message],
    ) -> AsyncIterator[EitherStreamedResponse]:
        """Stream response from randomly selected model."""
        models = await self._initialize_models()
        selected = random.choice(models)
        logger.debug("Selected model: %s", selected)
        async with selected.request_stream(messages) as stream:
            yield stream


class RandomMultiModel(MultiModel):
    """Randomly selects from available models."""

    type: Literal["random"] = "random"

    models: list[KnownModelName | Model] = Field(min_length=1)
    """List of models to use."""

    @model_validator(mode="after")
    def validate_models(self) -> RandomMultiModel:
        """Validate model configuration."""
        if not self.models:
            msg = "At least one model must be provided"
            raise ValueError(msg)
        return self

    def name(self) -> str:
        """Get model name."""
        return f"multi-random({len(self.models)})"

    @property
    def available_models(self) -> list[Model]:
        """Get list of available models."""
        return [
            model if isinstance(model, Model) else infer_model(model)  # type: ignore[arg-type]
            for model in self.models
        ]

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create random agent model."""
        return RandomAgentModel(
            models=self.available_models,
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )
