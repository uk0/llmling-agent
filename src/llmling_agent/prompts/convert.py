from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, Literal, Protocol, TypeVar

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


class Template(str):  # noqa: SLOT000
    """Marker class for strings that should be templated."""


async def to_prompt(  # noqa: PLR0911
    obj: AnyPromptType,
    *,
    template_mode: Literal["off", "explicit", "all"] = "off",
    **kwargs: Any,
) -> str:
    """Convert any supported type to a prompt string.

    Args:
        obj: Object to convert
        template_mode: How to handle templating:
            - "off": No templating (default)
            - "explicit": Only Template instances
            - "all": Template all strings (dangerous!)
        **kwargs: Template variables if templating is enabled
    """
    match obj:
        case Template() if template_mode != "off":
            return render_prompt(obj, kwargs)

        case str() if template_mode == "all":
            return render_prompt(obj, kwargs)

        case str():
            return obj
        case type() if hasattr(obj, "__prompt_type__"):
            return obj.__prompt_type__()

        case _ if hasattr(obj, "__prompt__"):
            return obj.__prompt__()  # pyright: ignore[reportAttributeAccessIssue]

        case BasePrompt():
            messages = await obj.format(kwargs)
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
            return await to_prompt(result, **kwargs)

        case _ if callable(obj):
            result = obj()
            return await to_prompt(result, **kwargs)

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


def render_prompt(
    template: str,
    agent_context: dict[str, Any],
) -> str:
    """Render a prompt template with context.

    Available variables:
        agent.name: Name of the agent
        agent.id: Number of the clone (for cloned agents)
        agent.model: Model name
    """
    from jinja2 import Environment

    env = Environment(autoescape=True, keep_trailing_newline=True)
    tpl = env.from_string(template)
    return tpl.render(agent=agent_context)


def can_format_fields(obj: Any) -> bool:
    """Check if object can be inspected by fieldz."""
    try:
        get_adapter(obj)
    except TypeError:
        return False
    else:
        return True
