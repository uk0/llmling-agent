"""Response utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pydantic import BaseModel

    from llmling_agent.models import AgentContext
    from llmling_agent.responses.models import InlineResponseDefinition


def resolve_response_type(
    type_name: str | InlineResponseDefinition,
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

    match type_name:
        case str() if context and type_name in context.definition.responses:
            # Get from shared responses
            response_def = context.definition.responses[type_name]
            match response_def:
                case ImportedResponseDefinition():
                    return response_def.resolve_model()
                case InlineResponseDefinition():
                    return response_def.create_model()
        case InlineResponseDefinition():
            # Handle inline definition
            return type_name.create_model()
        case _:
            msg = f"Invalid result type: {type_name}"
            raise ValueError(msg)
