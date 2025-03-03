"""Type definitions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import os
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID

from llmling import LLMCallableTool


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for model objects."""

    @property
    def model_name(self) -> str: ...


# Define what we consider JSON-serializable
type JsonPrimitive = None | bool | int | float | str
type JsonValue = JsonPrimitive | JsonArray | JsonObject
type JsonObject = dict[str, JsonValue]
type JsonArray = list[JsonValue]

# In reflex for example, the complex ones create issues..
SimpleJsonType = dict[
    str, bool | int | float | str | list[str] | dict[str, bool | int | float | str]
]
type StrPath = str | os.PathLike[str]
type SessionIdType = str | UUID | None


NodeName = str
TeamName = str
AgentName = str
MessageRole = Literal["user", "assistant", "system"]
PartType = Literal["text", "image", "audio", "video"]
ModelType = ModelProtocol | str | None
EnvironmentType = Literal["file", "inline"]
ToolSource = Literal["runtime", "agent", "builtin", "dynamic", "task", "mcp", "toolset"]
AnyCallable = Callable[..., Any]
AsyncFilterFn = Callable[..., Awaitable[bool]]
SyncFilterFn = Callable[..., bool]
AnyFilterFn = Callable[..., bool | Awaitable[bool]]
type AnyTransformFn[T] = Callable[[T], T | Awaitable[T]]
type OptionalAwaitable[T] = T | Awaitable[T]

type ToolType = str | AnyCallable | LLMCallableTool

# P = ParamSpec("P")
# SyncAsync = Callable[P, OptionalAwaitable[T]]
EndStrategy = Literal["early", "exhaustive"]
QueueStrategy = Literal["concat", "latest", "buffer"]
"""The strategy for handling multiple tool calls when a final result is found.

- `'early'`: Stop processing other tool calls once a final result is found
- `'exhaustive'`: Process all tool calls even after finding a final result
"""
