"""Decorator to add sync() method to async functions."""

from __future__ import annotations

import asyncio
import concurrent.futures
from functools import wraps
import inspect
from typing import TYPE_CHECKING, Any, Concatenate
import warnings


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class AsyncSyncWrapper[**P, T]:
    """Wrapper that provides both async __call__ and sync() methods."""

    def __init__(self, func: Callable[Concatenate[Any, P], Awaitable[T]]):
        self._func = func
        self._instance: Any = None
        # Copy function metadata
        wraps(func)(self)

    def __get__(self, instance: Any, owner: type | None = None) -> AsyncSyncWrapper[P, T]:
        """Descriptor protocol for method binding."""
        if instance is None:
            return self
        # Create bound wrapper
        bound = type(self)(self._func)
        bound._instance = instance
        return bound

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Async call - normal behavior."""
        if self._instance is not None:
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


def add_sync[**P, T](
    func: Callable[Concatenate[Any, P], Awaitable[T]],
) -> AsyncSyncWrapper[P, T]:
    """Decorator to add sync() method to async functions.

    Usage:
        @add_sync
        async def my_func(x: int) -> str:
            return str(x)

        # Both work:
        result = await my_func(42)
        result = my_func.sync(42)
    """
    if not inspect.iscoroutinefunction(func):
        msg = f"@add_sync can only be applied to async functions, got {func}"
        raise TypeError(msg)

    return AsyncSyncWrapper(func)
