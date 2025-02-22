"""LLMling-models package."""

from typing import Annotated

from pydantic import Field
from llmling_agent_models.configs import (
    AISuiteModelConfig,
    AugmentedModelConfig,
    CostOptimizedModelConfig,
    DelegationModelConfig,
    FallbackModelConfig,
    ImportModelConfig,
    InputModelConfig,
    LLMAdapterConfig,
    RemoteInputConfig,
    RemoteProxyConfig,
    TokenOptimizedModelConfig,
    UserSelectModelConfig,
    PrePostPromptConfig,
    StringModelConfig,
    TestModelConfig,
)
from llmling_agent_models.base import BaseModelConfig

AnyModelConfig = Annotated[
    AISuiteModelConfig
    | AugmentedModelConfig
    | CostOptimizedModelConfig
    | DelegationModelConfig
    | FallbackModelConfig
    | ImportModelConfig
    | InputModelConfig
    | LLMAdapterConfig
    | RemoteInputConfig
    | RemoteProxyConfig
    | TokenOptimizedModelConfig
    | StringModelConfig
    | TestModelConfig
    | UserSelectModelConfig,
    Field(discriminator="type"),
]


__all__ = [
    "AISuiteModelConfig",
    "AnyModelConfig",
    "AugmentedModelConfig",
    "BaseModelConfig",
    "CostOptimizedModelConfig",
    "DelegationModelConfig",
    "FallbackModelConfig",
    "ImportModelConfig",
    "InputModelConfig",
    "LLMAdapterConfig",
    "PrePostPromptConfig",
    "RemoteInputConfig",
    "RemoteProxyConfig",
    "StringModelConfig",
    "TestModelConfig",
    "TokenOptimizedModelConfig",
    "UserSelectModelConfig",
]
