"""High-level functional interfaces for LLMling agent."""

from llmling_agent_functional.auto_generate import auto_callable
from llmling_agent_functional.run import (
    run_agent,
    run_agent_sync,
)
from llmling_agent_functional.structure import (
    get_structured,
    get_structured_multiple,
    pick_one,
)

__all__ = [
    "auto_callable",
    "get_structured",
    "get_structured_multiple",
    "pick_one",
    "run_agent",
    "run_agent_sync",
]
