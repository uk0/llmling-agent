"""Configuration models for LLMling models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ImportString, SecretStr

from llmling_agent_models.base import BaseModelConfig


class AISuiteModelConfig(BaseModelConfig):
    """Configuration for AISuite adapter."""

    type: Literal["aisuite"] = Field(default="aisuite", init=False)
    model: str
    config: dict[str, dict[str, str]] = Field(default_factory=dict)

    def get_model(self) -> Any:
        from llmling_models.aisuite_adapter import AISuiteAdapter

        return AISuiteAdapter(model=self.model, config=self.config)


class PrePostPromptConfig(BaseModel):
    """Configuration for pre/post prompts."""

    text: str
    model: str


class AugmentedModelConfig(BaseModelConfig):
    """Configuration for model with pre/post prompt processing."""

    type: Literal["augmented"] = Field(default="augmented", init=False)
    main_model: str
    pre_prompt: PrePostPromptConfig | None = None
    post_prompt: PrePostPromptConfig | None = None

    def get_model(self) -> Any:
        from llmling_models.augmented import AugmentedModel

        return AugmentedModel(
            main_model=self.main_model,  # type: ignore
            pre_prompt=self.pre_prompt,  # type: ignore
            post_prompt=self.post_prompt,  # type: ignore
        )


class CostOptimizedModelConfig(BaseModelConfig):
    """Configuration for cost-optimized model selection."""

    type: Literal["cost-optimized"] = Field(default="cost-optimized", init=False)
    models: list[str | BaseModelConfig]
    max_input_cost: float = Field(gt=0)
    strategy: Literal["cheapest_possible", "best_within_budget"] = "best_within_budget"

    def get_model(self) -> Any:
        from llmling_models.multimodels import CostOptimizedMultiModel

        converted_models = [
            m.get_model() if isinstance(m, BaseModelConfig) else m for m in self.models
        ]
        return CostOptimizedMultiModel(
            models=converted_models,
            max_input_cost=self.max_input_cost,
            strategy=self.strategy,
        )


class DelegationModelConfig(BaseModelConfig):
    """Configuration for delegation-based model selection."""

    type: Literal["delegation"] = Field(default="delegation", init=False)
    selector_model: str | BaseModelConfig
    models: list[str | BaseModelConfig]
    selection_prompt: str
    model_descriptions: dict[str, str] | None = (
        None  # Optional descriptions as separate field
    )

    def get_model(self) -> Any:
        from llmling_models.multimodels import DelegationMultiModel

        # Convert selector if it's a config
        selector = (
            self.selector_model.get_model()
            if isinstance(self.selector_model, BaseModelConfig)
            else self.selector_model
        )

        # Convert model list
        converted_models = [
            m.get_model() if isinstance(m, BaseModelConfig) else m for m in self.models
        ]

        return DelegationMultiModel(
            selector_model=selector,
            models=converted_models,
            selection_prompt=self.selection_prompt,
        )


class FallbackModelConfig(BaseModelConfig):
    """Configuration for fallback strategy."""

    type: Literal["fallback"] = Field(default="fallback", init=False)
    models: list[str | BaseModelConfig]

    def get_model(self) -> Any:
        from llmling_models.multimodels import FallbackMultiModel

        # Convert nested configs to models
        converted_models = [
            model.get_model() if isinstance(model, BaseModelConfig) else model
            for model in self.models
        ]
        return FallbackMultiModel(models=converted_models)


class ImportModelConfig(BaseModelConfig):
    """Configuration for importing external models."""

    type: Literal["import"] = Field(default="import", init=False)
    model: ImportString
    kw_args: dict[str, str] = Field(default_factory=dict)

    def get_model(self) -> Any:
        return self.model(**self.kw_args) if isinstance(self.model, type) else self.model


class InputModelConfig(BaseModelConfig):
    """Configuration for human input model."""

    type: Literal["input"] = Field(default="input", init=False)
    prompt_template: str = Field(default="👤 Please respond to: {prompt}")
    show_system: bool = Field(default=True)
    input_prompt: str = Field(default="Your response: ")
    handler: ImportString = Field(
        default="llmling_models:DefaultInputHandler",
        validate_default=True,
    )

    def get_model(self) -> Any:
        from llmling_models.inputmodel import InputModel

        return InputModel(
            prompt_template=self.prompt_template,
            show_system=self.show_system,
            input_prompt=self.input_prompt,
            handler=self.handler,
        )


class LLMAdapterConfig(BaseModelConfig):
    """Configuration for LLM library adapter."""

    type: Literal["llm"] = Field(default="llm", init=False)
    model_name: str

    def get_model(self) -> Any:
        from llmling_models.llm_adapter import LLMAdapter

        return LLMAdapter(model_name=self.model_name)


class RemoteInputConfig(BaseModelConfig):
    """Configuration for remote human input."""

    type: Literal["remote-input"] = Field(default="remote-input", init=False)
    url: str = "ws://localhost:8000/v1/chat/stream"
    api_key: SecretStr | None = None

    def get_model(self) -> Any:
        from llmling_models.remote_input.client import RemoteInputModel

        key = self.api_key.get_secret_value() if self.api_key else None
        return RemoteInputModel(url=self.url, api_key=key)


class RemoteProxyConfig(BaseModelConfig):
    """Configuration for remote model proxy."""

    type: Literal["remote-proxy"] = Field(default="remote-proxy", init=False)
    url: str = "ws://localhost:8000/v1/completion/stream"
    api_key: SecretStr | None = None

    def get_model(self) -> Any:
        from llmling_models.remote_model.client import RemoteProxyModel

        key = self.api_key.get_secret_value() if self.api_key else None
        return RemoteProxyModel(url=self.url, api_key=key)


class TokenOptimizedModelConfig(BaseModelConfig):
    """Configuration for token-optimized model selection."""

    type: Literal["token-optimized"] = Field(default="token-optimized", init=False)
    models: list[str | BaseModelConfig]
    strategy: Literal["efficient", "maximum_context"] = Field(default="efficient")

    def get_model(self) -> Any:
        from llmling_models.multimodels import TokenOptimizedMultiModel

        converted_models = [
            m.get_model() if isinstance(m, BaseModelConfig) else m for m in self.models
        ]
        return TokenOptimizedMultiModel(
            models=converted_models,
            strategy=self.strategy,
        )


class UserSelectModelConfig(BaseModelConfig):
    """Configuration for interactive model selection."""

    type: Literal["user-select"] = Field(default="user-select", init=False)
    models: list[str | BaseModelConfig]
    prompt_template: str = Field(default="🤖 Choose a model for: {prompt}")
    show_system: bool = Field(default=True)
    input_prompt: str = Field(default="Enter model number (0-{max}): ")
    handler: ImportString = Field(
        default="llmling_models:DefaultInputHandler",
        validate_default=True,
    )

    def get_model(self) -> Any:
        from llmling_models.multimodels import UserSelectModel

        converted_models = [
            m.get_model() if isinstance(m, BaseModelConfig) else m for m in self.models
        ]
        return UserSelectModel(
            models=converted_models,
            prompt_template=self.prompt_template,
            show_system=self.show_system,
            input_prompt=self.input_prompt,
            handler=self.handler,
        )


class StringModelConfig(BaseModelConfig):
    """Configuration for string-based model references."""

    type: Literal["string"] = Field(default="string", init=False)
    identifier: str

    def get_model(self) -> Any:
        # from llmling_models import infer_model
        # return infer_model(self.identifier)
        from llmling_models.model_types import StringModel

        return StringModel(identifier=self.identifier)


class TestModelConfig(BaseModelConfig):
    """Configuration for test models."""

    type: Literal["test"] = Field(default="test", init=False)
    custom_result_text: str | None = None
    call_tools: list[str] | Literal["all"] = "all"

    def get_model(self) -> Any:
        from pydantic_ai.models.test import TestModel

        return TestModel(
            custom_result_text=self.custom_result_text,
            call_tools=self.call_tools,
        )
