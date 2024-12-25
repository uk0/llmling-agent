"""High-level functional interfaces for LLMling agent."""

from llmling_agent.functional.auto_generate import auto_callable
from llmling_agent.functional.run import (
    run_agent_pipeline,
    run_agent_pipeline_sync,
    run_with_model,
    run_with_model_sync,
)
from llmling_agent.functional.structure import (
    get_structured,
    get_structured_multiple,
    pick_one,
)

__all__ = [
    "auto_callable",
    "get_structured",
    "get_structured_multiple",
    "pick_one",
    "run_agent_pipeline",
    "run_agent_pipeline_sync",
    "run_with_model",
    "run_with_model_sync",
]
