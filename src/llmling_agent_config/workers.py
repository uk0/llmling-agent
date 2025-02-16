"""Models for toolsets."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class WorkerConfig(BaseModel):
    """Configuration for a worker agent.

    Worker agents are agents that are registered as tools with a parent agent.
    This allows building hierarchies and specializations of agents.
    """

    name: str
    """Name of the agent to use as a worker"""

    reset_history_on_run: bool = True
    """Whether to clear worker's conversation history before each run.
    True (default): Fresh conversation each time
    False: Maintain conversation context between runs"""

    pass_message_history: bool = False
    """Whether to pass parent agent's message history to worker.
    True: Worker sees parent's conversation context
    False (default): Worker only sees current request"""

    share_context: bool = False
    """Whether to share parent agent's context/dependencies with worker.
    True: Worker has access to parent's context data
    False (default): Worker uses own isolated context"""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")
