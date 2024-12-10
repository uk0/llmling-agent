"""Factory for creating Pydantic-AI agents from configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from llmling.core import exceptions
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, create_model

from llmling_agent.agent import LLMlingAgent
from llmling_agent.log import get_logger
from llmling_agent.models import (
    AgentDefinition,
    ResponseDefinition,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling.config.runtime import RuntimeConfig

    from llmling_agent.models import (
        AgentConfig,
        AgentDefinition,
        ResponseDefinition,
    )


logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)

# Cache for created models to avoid recreation
_model_cache: dict[str, type[BaseModel]] = {}


def create_agents_from_config(
    config: AgentDefinition,
    runtime: RuntimeConfig,
) -> dict[str, LLMlingAgent[Any]]:
    """Create all agents from configuration.

    Args:
        config: Complete agent configuration
        runtime: Runtime configuration

    Returns:
        Dictionary mapping agent IDs to LLMling agents

    Raises:
        ConfigError: If configuration is invalid
    """
    agents = {}
    for agent_id, agent_config in config.agents.items():
        try:
            agents[agent_id] = _create_single_agent(
                agent_config,
                config.responses,
                runtime,
            )
        except Exception as exc:
            msg = f"Failed to create agent {agent_id}: {exc}"
            raise exceptions.ConfigError(msg) from exc
    return agents


def _create_single_agent(
    agent_config: AgentConfig,
    responses: dict[str, ResponseDefinition],
    runtime: RuntimeConfig,
) -> LLMlingAgent[Any]:
    """Internal helper to create a single agent."""
    # Get or create the result model
    result_type = _get_or_create_response_model(
        agent_config.result_type,
        responses[agent_config.result_type],
    )

    # Initialize type adapter to ensure model is built
    TypeAdapter(result_type).json_schema()

    return LLMlingAgent(
        runtime=runtime,
        result_type=result_type,
        model=agent_config.model,
        system_prompt=_create_system_prompts(agent_config.system_prompts),
        name=agent_config.name or "LLMling Agent",
        **agent_config.model_settings,
    )


def _get_or_create_response_model(
    name: str, definition: ResponseDefinition
) -> type[BaseModel]:
    """Get cached model or create new one."""
    if name not in _model_cache:
        _model_cache[name] = _create_response_model(name, definition)
    return _model_cache[name]


def _create_response_model(name: str, definition: ResponseDefinition) -> type[BaseModel]:
    """Create a Pydantic model from response definition."""
    fields: dict[str, Any] = {}

    for field_name, field_def in definition.fields.items():
        # Convert string type to actual type
        type_hint = _parse_type_annotation(field_def.type)

        # Create field with metadata
        field = Field(
            description=field_def.description,
            **(field_def.constraints or {}),
        )
        fields[field_name] = (type_hint, field)

    model = create_model(
        name,
        __config__=ConfigDict(frozen=True),
        __doc__=definition.description,
        **fields,
    )

    # Ensure model is fully built
    TypeAdapter(model).json_schema()

    return model


def _create_system_prompts(prompts: list[str]) -> Sequence[str]:
    """Convert system prompt configs to actual prompts."""
    return prompts


def _parse_type_annotation(type_str: str) -> Any:
    """Convert string type annotation to actual type."""
    import typing

    # Handle basic types
    basic_types = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "dict": dict,
        "list": list,
    }

    if type_str in basic_types:
        return basic_types[type_str]

    # Handle generic types (e.g. list[str])
    if "[" in type_str:
        container, inner = type_str.split("[", 1)
        inner = inner.rstrip("]")

        container_type = getattr(typing, container.capitalize())
        inner_type = _parse_type_annotation(inner)

        return container_type[inner_type]

    msg = f"Unknown type: {type_str}"
    raise ValueError(msg)
