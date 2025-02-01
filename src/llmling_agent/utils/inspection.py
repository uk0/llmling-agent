from __future__ import annotations

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
    from llmling_agent.common_types import AnyCallable


T = TypeVar("T")


def has_argument_type(func: AnyCallable, arg_type: str | type) -> bool:
    """Checks whether any argument of func is of type arg_type."""
    sig = inspect.signature(func)
    arg_str = arg_type if isinstance(arg_type, str) else arg_type.__name__
    return any(arg_str in str(param.annotation) for param in sig.parameters.values())


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
