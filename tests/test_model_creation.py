from __future__ import annotations

import typing

from pydantic import BaseModel
import pytest

from llmling_agent.factory import _create_response_model, _parse_type_annotation
from llmling_agent.models import ResponseDefinition, ResponseField


def test_parse_type_annotation() -> None:
    """Test string type parsing."""
    assert _parse_type_annotation("str") is str
    assert _parse_type_annotation("int") is int
    assert _parse_type_annotation("list[str]") == typing.List[str]  # noqa: UP006

    with pytest.raises(ValueError, match="invalid_type"):
        _parse_type_annotation("invalid_type")


def test_create_response_model() -> None:
    """Test dynamic model creation with proper field validation."""
    con = {"ge": 0, "le": 100}
    fields = {
        "message": ResponseField(type="str", description="A test message"),
        "count": ResponseField(type="int", constraints=con, description="A count value"),
        "items": ResponseField(type="list[str]", description="List of items"),
    }
    definition = ResponseDefinition(description="Test response", fields=fields)

    model = _create_response_model("TestResponse", definition)
    assert issubclass(model, BaseModel)
    assert model.__doc__ == "Test response"

    # Test field definitions
    message_field = model.model_fields["message"]
    assert message_field.annotation is str
    assert message_field.description == "A test message"
    assert not message_field.metadata  # No constraints

    # Test numeric constraints in metadata
    count_field = model.model_fields["count"]
    assert count_field.annotation is int
    assert count_field.description == "A count value"

    # Test list field
    items_field = model.model_fields["items"]
    assert items_field.annotation == typing.List[str]  # noqa: UP006
    assert items_field.description == "List of items"

    # Test validation works
    instance = model(message="test", count=50, items=["a", "b"])
    assert instance.message == "test"  # type: ignore
    assert instance.count == 50  # type: ignore  # noqa
    assert instance.items == ["a", "b"]  # type: ignore
