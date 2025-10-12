"""Client ACP Connection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from acp.schema import ModelInfo, SessionModelState


if TYPE_CHECKING:
    from collections.abc import Sequence

    from tokonomics.model_discovery.model_info import ModelInfo as TokoModelInfo


def create_session_model_state(
    available_models: Sequence[TokoModelInfo], current_model: str | None = None
) -> SessionModelState | None:
    """Create a SessionModelState from available models.

    Args:
        available_models: List of all models the agent can switch between
        current_model: The currently active model (defaults to first available)

    Returns:
        SessionModelState with all available models, None if no models provided
    """
    if not available_models:
        return None
    # Create ModelInfo objects for each available model
    models = [
        ModelInfo(
            model_id=model.pydantic_ai_id,
            name=f"{model.provider}: {model.name}",
            description=model.format(),
        )
        for model in available_models
    ]
    # Use first model as current if not specified
    all_ids = [model.pydantic_ai_id for model in available_models]
    current_model_id = current_model if current_model in all_ids else all_ids[0]
    return SessionModelState(available_models=models, current_model_id=current_model_id)
