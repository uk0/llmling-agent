"""Multi-model implementations."""

from __future__ import annotations

from contextlib import asynccontextmanager
import random
from typing import TYPE_CHECKING, Literal

from pydantic_ai.models import AgentModel, KnownModelName, Model, infer_model

from llmling_agent.log import get_logger
from llmling_agent.pydanticai_models.base import PydanticModel


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from pydantic_ai.messages import Message, ModelAnyResponse
    from pydantic_ai.models import EitherStreamedResponse
    from pydantic_ai.result import Cost
    from pydantic_ai.tools import ToolDefinition

    from llmling_agent.pydanticai_models.types import ModelInput


logger = get_logger(__name__)


class MultiAgentModel(AgentModel):
    """AgentModel implementation for multi-model setups."""

    def __init__(
        self,
        models: Sequence[ModelInput],
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> None:
        """Initialize multi-agent model.

        Args:
            models: List of models to use
            function_tools: Tools available for function calls
            allow_text_result: Whether text results are allowed
            result_tools: Tools for result validation
        """
        self.model_inputs = models
        self.function_tools = function_tools
        self.allow_text_result = allow_text_result
        self.result_tools = result_tools
        self._initialized_models: list[AgentModel] | None = None

    async def _initialize_models(self) -> list[AgentModel]:
        """Initialize all agent models."""
        if self._initialized_models is None:
            # Convert strings to Model instances
            models = [
                m if isinstance(m, Model) else infer_model(m)  # type: ignore
                for m in self.model_inputs
            ]
            # Initialize agent models
            self._initialized_models = []
            for model in models:
                agent_model = await model.agent_model(
                    function_tools=self.function_tools,
                    allow_text_result=self.allow_text_result,
                    result_tools=self.result_tools,
                )
                self._initialized_models.append(agent_model)
        return self._initialized_models

    async def request(
        self,
        messages: list[Message],
    ) -> tuple[ModelAnyResponse, Cost]:
        """Make request to selected model."""
        raise NotImplementedError

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[Message],
    ) -> AsyncIterator[EitherStreamedResponse]:
        """Stream response from selected model."""
        raise NotImplementedError
        yield  # required for generator


class MultiModel(PydanticModel):
    """Base for model configurations that combine multiple language models.

    Provides infrastructure for using multiple models in a coordinated way.
    Subclasses implement specific strategies for model selection and usage,
    such as random selection, round-robin, or conditional routing.
    """

    type: str
    """Discriminator field for multi-model types."""

    models: list[KnownModelName]
    """List of models to use."""

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
        return MultiAgentModel(
            models=self.models,
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )


class RandomAgentModel(MultiAgentModel):
    """Randomly selects from available models for each request."""

    async def request(
        self,
        messages: list[Message],
    ) -> tuple[ModelAnyResponse, Cost]:
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

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create random agent model."""
        return RandomAgentModel(
            models=self.models,
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )
