from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llmling import LLMCallableTool


# Define what we consider JSON-serializable
type JsonPrimitive = None | bool | int | float | str
type JsonValue = JsonPrimitive | JsonArray | JsonObject
type JsonObject = dict[str, JsonValue]
type JsonArray = list[JsonValue]

type ToolType = str | Callable[..., Any] | LLMCallableTool
