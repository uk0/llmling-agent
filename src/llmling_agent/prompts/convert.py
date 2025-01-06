from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, Protocol, TypeVar

from fieldz import fields, get_adapter
from llmling import BasePrompt
from pydantic import BaseModel


T = TypeVar("T")


class PromptConvertible(Protocol):
    """Protocol for instances that can be converted to prompts."""

    def __prompt__(self) -> str: ...


class PromptTypeConvertible(Protocol):
    """Protocol for types that can be converted to prompts."""

    @classmethod
    def __prompt_type__(cls) -> str: ...


class FieldFormattable(Protocol):
    """Protocol for types that can be formatted through their fields."""

    __annotations__: dict[str, Any]


type AnyPromptType = (
    str
    | PromptConvertible
    | PromptTypeConvertible
    | FieldFormattable
    | BaseModel
    | BasePrompt
    | dict[str, Any]
    | list[Any]
    | tuple[Any, ...]
    | Callable[..., str]
    | Coroutine[Any, Any, str]
)


async def to_prompt(obj: AnyPromptType, **format_kwargs: Any) -> str:  # noqa: PLR0911
    """Convert any supported type to a prompt string.

    Args:
        obj: Object to convert
        format_kwargs: Optional kwargs for prompt formatting

    Examples:
        >>> to_prompt("Hello")
        'Hello'

        >>> class Greeter:
        ...     def __prompt__(self) -> str:
        ...         return "Hello!"
        ...     @classmethod
        ...     def __prompt_type__(cls) -> str:
        ...         return "Greeter class that says hello"
        >>> to_prompt(Greeter())  # Instance prompt
        'Hello!'
        >>> to_prompt(Greeter)    # Type prompt
        'Greeter class that says hello'
    """
    match obj:
        case str():
            return obj

        case type() if hasattr(obj, "__prompt_type__"):
            return obj.__prompt_type__()

        case _ if hasattr(obj, "__prompt__"):
            return obj.__prompt__()  # pyright: ignore[reportAttributeAccessIssue]

        case BasePrompt():
            messages = await obj.format(format_kwargs)
            return "\n".join(msg.get_text_content() for msg in messages)

        case _ if can_format_fields(obj):
            return format_dataclass_like(obj)

        case dict():
            results = await asyncio.gather(*(to_prompt(v) for k, v in obj.items()))
            return "\n".join(f"{k}: {r}" for (k, _), r in zip(obj.items(), results))

        case list() | tuple():
            items = await asyncio.gather(*(to_prompt(item) for item in obj))
            return "\n".join(items)

        case Coroutine():
            result = await obj
            return await to_prompt(result, **format_kwargs)

        case _ if callable(obj):
            result = obj()
            return await to_prompt(result, **format_kwargs)

        case _:
            return str(obj)


def format_dataclass_like(obj: Any) -> str:
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
