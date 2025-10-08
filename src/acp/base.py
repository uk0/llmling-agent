"""Base class for generated models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


def convert(text: str):
    if text == "field_meta":
        return "_meta"
    return to_camel(text)


class Schema(BaseModel):
    """Base class for generated models."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=convert)
