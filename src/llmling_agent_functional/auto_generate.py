"""Auto-generate LLM responses using function signatures."""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

from llmling_agent_functional.structure import get_structured


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from pydantic_ai.agent import models

    from llmling_agent.common_types import AnyCallable


P = ParamSpec("P")
R = TypeVar("R")


# We need separate overloads for async and sync functions
@overload
def auto_callable(
    model: str | models.Model | models.KnownModelName,
    *,
    system_prompt: str | None = None,
    retries: int = 3,
) -> Callable[
    [Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]
]: ...


@overload
def auto_callable(
    model: str | models.Model | models.KnownModelName,
    *,
    system_prompt: str | None = None,
    retries: int = 3,
) -> Callable[[Callable[P, R]], Callable[P, Coroutine[Any, Any, R]]]: ...


def auto_callable(
    model: str | models.Model | models.KnownModelName,
    *,
    system_prompt: str | None = None,
    retries: int = 3,
) -> AnyCallable:
    """Use function signature as schema for LLM responses.

    This decorator uses the function's:
    - Type hints
    - Docstring
    - Parameter names and defaults
    as a schema for getting structured responses from the LLM.

    Args:
        model: Model to use for responses
        system_prompt: Optional system instructions
        retries: Max retries for failed responses

    Example:
        @auto_callable("gpt-4")
        async def analyze_sentiment(text: str) -> dict[str, float]:
            '''Analyze sentiment scores (positive/negative) for text.'''
    """

    def decorator(
        func: Callable[P, R] | Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        # Get function info once
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Bind arguments to create context
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            arg_values = dict(bound.arguments)

            # Create prompt from signature and args
            prompt = (
                f"Based on this function:\n\n"
                f"def {func.__name__}{sig!s}:\n"
                f'    """{doc}"""\n\n'
                f"Generate response for inputs: {arg_values}"
            )

            # Use get_structured to get typed response
            return_type = sig.return_annotation
            if return_type is inspect.Parameter.empty:
                return_type = str  # type: ignore[assignment]

            return await get_structured(
                prompt=prompt,
                response_type=return_type,  # type: ignore[arg-type]
                model=model,
                system_prompt=system_prompt,
                max_retries=retries,
            )

        return wrapper

    return decorator
