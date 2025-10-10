"""Progress notification handler type for MCP to external system bridging."""

from __future__ import annotations

from collections.abc import Awaitable, Callable


# Simple callable type for handling MCP progress notifications
ProgressHandler = Callable[
    [
        str,
        str,
        dict,
        float,
        float | None,
        str | None,
    ],  # tool_name, tool_call_id, tool_input, progress, total, message
    Awaitable[None],
]
