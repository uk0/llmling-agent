"""Factory for creating Pydantic-AI agents from configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, create_model

from llmling_agent.log import get_logger
from llmling_agent.models import (
    ResponseDefinition,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling_agent.models import (
        ResponseDefinition,
    )


logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)

# Cache for created models to avoid recreation
_model_cache: dict[str, type[BaseModel]] = {}


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
