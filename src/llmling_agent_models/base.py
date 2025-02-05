from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BaseModelConfig(BaseModel):
    """Base for model configurations."""

    type: str = Field(init=False)
    """Type discriminator for model configs."""

    def get_model(self) -> Any:
        """Create and return actual model instance."""
        msg = f"Model creation not implemented for {self.__class__.__name__}"
        raise NotImplementedError(msg)
