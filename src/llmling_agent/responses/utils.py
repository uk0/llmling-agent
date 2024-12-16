"""Response utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, create_model


if TYPE_CHECKING:
    from llmling_agent.context import AgentContext


TYPE_MAP = {
    "str": str,
    "bool": bool,
    "int": int,
    "float": float,
    "list[str]": list[str],
}


def resolve_response_type(
    type_name: str,
    context: AgentContext | None,
) -> type[BaseModel]:
    """Resolve response type from string name to actual type.

    Args:
        type_name: Name of the response type
        context: Agent context containing response definitions

    Returns:
        Resolved Pydantic model type

    Raises:
        ValueError: If type cannot be resolved
    """
    from llmling_agent.responses import (
        ImportedResponseDefinition,
        InlineResponseDefinition,
    )

    if not context or type_name not in context.definition.responses:
        msg = f"Result type {type_name} not found in responses"
        raise ValueError(msg)

    response_def = context.definition.responses[type_name]
    match response_def:
        case ImportedResponseDefinition():
            return response_def.resolve_model()
        case InlineResponseDefinition():
            # Create Pydantic model from inline definition
            fields = {}
            for name, field in response_def.fields.items():
                python_type = TYPE_MAP.get(field.type)
                if not python_type:
                    msg = f"Unsupported field type: {field.type}"
                    raise ValueError(msg)

                field_info = Field(description=field.description)
                fields[name] = (python_type, field_info)
            cls_name = response_def.description or "ResponseType"
            return create_model(cls_name, **fields, __base__=BaseModel)  # type: ignore[call-overload]
        case _:
            msg = f"Unknown response definition type: {type(response_def)}"
            raise ValueError(msg)
