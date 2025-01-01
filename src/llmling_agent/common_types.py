from __future__ import annotations

from collections.abc import Callable
import os
from typing import Any, Literal

from llmling import LLMCallableTool


# Define what we consider JSON-serializable
type JsonPrimitive = None | bool | int | float | str
type JsonValue = JsonPrimitive | JsonArray | JsonObject
type JsonObject = dict[str, JsonValue]
type JsonArray = list[JsonValue]

type ToolType = str | Callable[..., Any] | LLMCallableTool

type StrPath = str | os.PathLike[str]
type PromptFunction = Callable[..., str]

MessageRole = Literal["user", "assistant", "system"]
PartType = Literal["text", "image", "audio", "video"]

EnvironmentType = Literal["file", "inline"]
