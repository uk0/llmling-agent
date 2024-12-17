"""Models for response fields and definitions."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from llmling.utils import importing
from pydantic import BaseModel, ConfigDict, Field, create_model


TYPE_MAP = {
    "str": str,
    "bool": bool,
    "int": int,
    "float": float,
    "list[str]": list[str],
}


class ResponseField(BaseModel):
    """Field definition for inline response types.

    Defines a single field in an inline response definition, including:
    - Data type specification
    - Optional description
    - Validation constraints

    Used by InlineResponseDefinition to structure response fields.
    """

    type: str
    """Data type of the response field"""
    description: str | None = None
    """Optional description of what this field represents"""
    constraints: dict[str, Any] | None = None
    """Optional validation constraints for the field"""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")


class InlineResponseDefinition(BaseModel):
    """Inline definition of an agent's response structure.

    Allows defining response types directly in the configuration using:
    - Field definitions with types and descriptions
    - Optional validation constraints
    - Custom field descriptions

    Example:
        responses:
          BasicResult:
            type: inline
            fields:
              success: {type: bool, description: "Operation success"}
              message: {type: str, description: "Result details"}
    """

    type: Literal["inline"] = Field("inline", init=False)
    description: str | None = None
    fields: dict[str, ResponseField]
    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    def create_model(self) -> type[BaseModel]:  # type: ignore
        """Create Pydantic model from inline definition."""
        fields = {}
        for name, field in self.fields.items():
            python_type = TYPE_MAP.get(field.type)
            if not python_type:
                msg = f"Unsupported field type: {field.type}"
                raise ValueError(msg)

            field_info = Field(description=field.description)
            fields[name] = (python_type, field_info)

        cls_name = self.description or "ResponseType"
        return create_model(cls_name, **fields, __base__=BaseModel)  # type: ignore[call-overload]


class ImportedResponseDefinition(BaseModel):
    """Response definition that imports an existing Pydantic model.

    Allows using externally defined Pydantic models as response types.
    Benefits:
    - Reuse existing model definitions
    - Full Python type support
    - Complex validation logic
    - IDE support for imported types

    Example:
        responses:
          AnalysisResult:
            type: import
            import_path: myapp.models.AnalysisResult
    """

    type: Literal["import"] = Field("import", init=False)
    description: str | None = None
    import_path: str
    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    # mypy is confused about
    def resolve_model(self) -> type[BaseModel]:  # type: ignore
        """Import and return the model class."""
        try:
            model_class = importing.import_class(self.import_path)
            if not issubclass(model_class, BaseModel):
                msg = f"{self.import_path} must be a Pydantic model"
                raise TypeError(msg)  # noqa: TRY301
        except Exception as e:
            msg = f"Failed to import response type {self.import_path}"
            raise ValueError(msg) from e
        else:
            return model_class


ResponseDefinition = Annotated[
    InlineResponseDefinition | ImportedResponseDefinition, Field(discriminator="type")
]
