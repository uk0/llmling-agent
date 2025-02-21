"""Helper functions for running examples in different environments."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from typing import TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Awaitable


T = TypeVar("T")


def is_pyodide() -> bool:
    """Check if we're running in Pyodide."""
    return sys.platform == "emscripten"


def get_config_path(module_path: str | None = None, filename: str = "config.yml") -> Path:
    """Get the configuration file path based on environment.

    Args:
        module_path: Optional __file__ from the calling module (ignored in Pyodide)
        filename: Name of the config file (default: config.yml)

    Returns:
        Path to the configuration file
    """
    if is_pyodide():
        return Path(filename)
    if module_path is None:
        msg = "module_path is required in non-Pyodide environment"
        raise ValueError(msg)
    return Path(module_path).parent / filename


def run[T](coro: Awaitable[T]) -> T:
    """Run a coroutine in both normal Python and Pyodide environments."""
    try:
        # Check if we're in an event loop
        asyncio.get_running_loop()
        # If we are, run until complete
        return asyncio.get_event_loop().run_until_complete(coro)
    except RuntimeError:
        # No running event loop, create one
        return asyncio.run(coro)  # type: ignore
