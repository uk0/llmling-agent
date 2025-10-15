"""Decorator to add sync() method to async functions."""

from __future__ import annotations

import asyncio
import concurrent.futures
from functools import wraps
import inspect
from typing import TYPE_CHECKING, Any, Concatenate, Literal, overload
import warnings


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class AsyncSyncWrapper[**P, T]:
    """Wrapper that provides both async __call__ and sync() methods."""

    def __init__(self, func: Callable[..., Awaitable[T]], *, is_bound: bool = False):
        self._func = func
        self._instance: Any = None
        self._is_bound = is_bound
        # Copy function metadata
        wraps(func)(self)

    def __get__(self, instance: Any, owner: type | None = None) -> AsyncSyncWrapper[P, T]:
        """Descriptor protocol for method binding."""
        if instance is None:
            return self
        # Create bound wrapper only if this is a bound method
        if self._is_bound:
            bound = type(self)(self._func, is_bound=True)
            bound._instance = instance
            return bound
        return self

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Async call - normal behavior."""
        if self._instance is not None and self._is_bound:
            # We're bound to an instance, prepend it to args
            return await self._func(self._instance, *args, **kwargs)
        return await self._func(*args, **kwargs)

    def sync(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Synchronous version using asyncio.run or thread pool."""
        coro = self(*args, **kwargs)

        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(coro)
        else:
            # We're in an async context, fall back to thread pool
            warnings.warn(
                "Calling .sync() from async context - using thread pool. "
                "Consider using 'await' instead for better performance.",
                UserWarning,
                stacklevel=2,
            )
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped function."""
        return getattr(self._func, name)


@overload
def add_sync[**P, T](
    func: Callable[P, Awaitable[T]], *, bound: Literal[False] = False
) -> AsyncSyncWrapper[P, T]: ...


@overload
def add_sync[**P, T](
    func: Callable[Concatenate[Any, P], Awaitable[T]], *, bound: Literal[True]
) -> AsyncSyncWrapper[P, T]: ...


@overload
def add_sync[**P, T](
    *, bound: Literal[False] = False
) -> Callable[[Callable[P, Awaitable[T]]], AsyncSyncWrapper[P, T]]: ...


@overload
def add_sync[**P, T](
    *, bound: Literal[True]
) -> Callable[[Callable[Concatenate[Any, P], Awaitable[T]]], AsyncSyncWrapper[P, T]]: ...


def add_sync[**P, T](
    func: Callable[P, Awaitable[T]]
    | Callable[Concatenate[Any, P], Awaitable[T]]
    | None = None,
    *,
    bound: bool = False,
) -> (
    AsyncSyncWrapper[P, T]
    | Callable[
        [Callable[P, Awaitable[T]] | Callable[Concatenate[Any, P], Awaitable[T]]],
        AsyncSyncWrapper[P, T],
    ]
):
    """Decorator to add sync() method to async functions.

    Usage:
        # For standalone functions
        @add_sync
        async def my_func(x: int) -> str:
            return str(x)

        # For methods - must specify bound=True
        class MyClass:
            @add_sync(bound=True)
            async def my_method(self, x: int) -> str:
                return str(x)

        # Both work:
        result = await my_func(42)
        result = my_func.sync(42)
        result = await obj.my_method(42)
        result = obj.my_method.sync(42)
    """

    def decorator(
        f: Callable[P, Awaitable[T]] | Callable[Concatenate[Any, P], Awaitable[T]],
    ) -> AsyncSyncWrapper[P, T]:
        if not inspect.iscoroutinefunction(f):
            msg = f"@add_sync can only be applied to async functions, got {f}"
            raise TypeError(msg)
        return AsyncSyncWrapper(f, is_bound=bound)

    if func is None:
        # Called as @add_sync(bound=True)
        return decorator
    # Called as @add_sync
    return decorator(func)
