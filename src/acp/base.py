from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Schema(BaseModel):
    """Base class for generated models."""

    model_config = ConfigDict(populate_by_name=True)
