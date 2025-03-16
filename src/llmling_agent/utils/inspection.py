from __future__ import annotations

import asyncio
from collections.abc import Sequence
from importlib.util import find_spec
import inspect
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAliasType,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from llmling_agent.agent import AgentContext


T = TypeVar("T")

PACKAGE_NAME = "llmling-agent"


async def execute[T](
    func: Callable[..., T | Awaitable[T]],
    *args: Any,
    use_thread: bool = False,
    **kwargs: Any,
) -> T:
    """Execute callable, handling both sync and async cases."""
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)

    if use_thread:
        return await asyncio.to_thread(func, *args, **kwargs)  # type: ignore

    return func(*args, **kwargs)  # type: ignore


def has_argument_type(
    func: Callable[..., Any],
    arg_type: type | str | UnionType | Sequence[type | str | UnionType],
    include_return: bool = False,
) -> bool:
    """Check if function has any argument of specified type(s).

    Args:
        func: Function to check
        arg_type: Type(s) to look for. Can be:
            - Single type (int, str, etc)
            - Union type (int | str)
            - Type name as string
            - Sequence of the above
        include_return: Whether to also check return type annotation

    Examples:
        >>> def func(x: int | str, y: list[int]): ...
        >>> has_argument_type(func, int | str)  # True
        >>> has_argument_type(func, int)        # True
        >>> has_argument_type(func, list)       # True
        >>> has_argument_type(func, float)      # False
        >>> has_argument_type(func, (int, str)) # True

    Returns:
        True if any argument matches any of the target types
    """
    # Convert target type(s) to set of normalized strings
    if isinstance(arg_type, Sequence) and not isinstance(arg_type, str | bytes):
        target_types = {_type_to_string(t) for t in arg_type}
    else:
        target_types = {_type_to_string(arg_type)}

    # Get type hints including return type if requested
    hints = get_type_hints(func, include_extras=True)
    if not include_return:
        hints.pop("return", None)

    # Check each parameter's type annotation
    for param_type in hints.values():
        # Handle type aliases
        if isinstance(param_type, TypeAliasType):
            param_type = param_type.__value__

        # Check for direct match
        if _type_to_string(param_type) in target_types:
            return True

        # Handle Union types (both | and Union[...])
        origin = get_origin(param_type)
        if origin is Union or origin is UnionType:
            union_members = get_args(param_type)
            # Check each union member
            if any(_type_to_string(t) in target_types for t in union_members):
                return True
            # Also check if the complete union type matches
            if _type_to_string(param_type) in target_types:
                return True

        # Handle generic types (list[str], dict[str, int], etc)
        if origin is not None:
            # Check if the generic type (e.g., list) matches
            if _type_to_string(origin) in target_types:
                return True
            # Check type arguments (e.g., str in list[str])
            args = get_args(param_type)
            if any(_type_to_string(arg) in target_types for arg in args):
                return True

    return False


def _type_to_string(type_hint: Any) -> str:
    """Convert type to normalized string representation for comparison."""
    match type_hint:
        case str():
            return type_hint
        case type():
            return type_hint.__name__
        case TypeAliasType():
            return _type_to_string(type_hint.__value__)
        case UnionType():
            args = get_args(type_hint)
            args_str = ", ".join(_type_to_string(t) for t in args)
            return f"Union[{args_str}]"
        case _:
            return str(type_hint)


def has_return_type[T](  # noqa: PLR0911
    func: Callable[..., Any],
    expected_type: type[T],
) -> TypeGuard[Callable[..., T | Awaitable[T]]]:
    """Check if a function has a specific return type annotation.

    Args:
        func: Function to check
        expected_type: The type to check for

    Returns:
        True if function returns the expected type (or Awaitable of it)
    """
    hints = get_type_hints(func)
    if "return" not in hints:
        return False

    return_type = hints["return"]

    # Handle direct match
    if return_type is expected_type:
        return True

    # Handle TypeAliases
    if isinstance(return_type, TypeAliasType):
        return_type = return_type.__value__

    # Handle Union types (including Optional)
    origin = get_origin(return_type)
    args = get_args(return_type)

    if origin is Union or origin is UnionType:
        # Check each union member
        def check_type(t: Any) -> bool:
            return has_return_type(lambda: t, expected_type)

        return any(check_type(arg) for arg in args)

    # Handle Awaitable/Coroutine types
    if origin is not None and inspect.iscoroutinefunction(func):
        # For async functions, check the first type argument
        if args:
            # Recursively check the awaited type
            return has_return_type(lambda: args[0], expected_type)
        return False

    # Handle generic types (like list[str], etc)
    if origin is not None:
        return origin is expected_type

    return False


def call_with_context(
    func: Callable[..., T],
    context: AgentContext[Any],
    **kwargs: Any,
) -> T:
    """Call function with appropriate context injection.

    Handles:
    - Simple functions
    - Bound methods
    - Functions expecting AgentContext
    - Functions expecting context data
    """
    from llmling_agent.agent import AgentContext

    if inspect.ismethod(func):
        if has_argument_type(func, AgentContext):
            return func(context)
        return func()
    if has_argument_type(func, AgentContext):
        return func(context, **kwargs)
    return func(context.data)


def validate_import(module_path: str, extras_name: str):
    """Check existence of module, showing helpful error if not installed."""
    if not find_spec(module_path):
        msg = f"""
Optional dependency {module_path!r} not found.
Install with: pip install {PACKAGE_NAME}[{extras_name}]
"""
        raise ImportError(msg.strip())
