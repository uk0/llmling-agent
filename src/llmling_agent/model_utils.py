from __future__ import annotations

from typing import Any

from fieldz import fields, get_adapter


def format_instance_for_llm(obj: Any) -> str:
    """Format object instance showing structure and current values."""
    try:
        obj_fields = fields(obj)
    except TypeError:
        return f"Unable to inspect fields of {type(obj)}"

    lines = [f"{type(obj).__name__}:\n{type(obj).__doc__}\n"]

    for field in obj_fields:
        if field.name.startswith("_"):
            continue
        value = getattr(obj, field.name)
        if field.description:
            lines.append(f"- {field.name} = {value!r} ({field.description})")
        else:
            type_name = field.type if field.type else "any"
            lines.append(f"- {field.name} = {value!r} ({type_name})")

    return "\n".join(lines)


def can_format_fields(obj: Any) -> bool:
    """Check if object can be inspected by fieldz."""
    try:
        get_adapter(obj)
    except TypeError:
        return False
    else:
        return True


if __name__ == "__main__":
    from pydantic import BaseModel, ConfigDict

    class Foo(BaseModel):
        """My dataclass."""

        a: int
        """Docstring for a."""
        b: str
        c: list[int]

        model_config = ConfigDict(use_attribute_docstrings=True)

    foo = Foo(a=1, b="2", c=[3, 4])
    print(format_instance_for_llm(foo))
