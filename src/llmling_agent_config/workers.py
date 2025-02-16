"""Worker configuration models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseWorkerConfig(BaseModel):
    """Base configuration for workers.

    Workers are nodes that can be registered as tools with a parent node.
    This allows building hierarchies of specialized nodes.
    """

    name: str
    """Name of the node to use as a worker."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")


class TeamWorkerConfig(BaseWorkerConfig):
    """Configuration for team workers.

    Team workers allow using entire teams as tools for other nodes.
    """

    type: Literal["team"] = Field("team", init=False)
    """Team worker configuration."""


class AgentWorkerConfig(BaseWorkerConfig):
    """Configuration for agent workers.

    Agent workers provide advanced features like history management and
    context sharing between agents.
    """

    type: Literal["agent"] = Field("agent", init=False)
    """Agent worker configuration."""

    reset_history_on_run: bool = True
    """Whether to clear worker's conversation history before each run.
    True (default): Fresh conversation each time
    False: Maintain conversation context between runs
    """

    pass_message_history: bool = False
    """Whether to pass parent agent's message history to worker.
    True: Worker sees parent's conversation context
    False (default): Worker only sees current request
    """

    share_context: bool = False
    """Whether to share parent agent's context/dependencies with worker.
    True: Worker has access to parent's context data
    False (default): Worker uses own isolated context
    """


WorkerConfig = Annotated[
    TeamWorkerConfig | AgentWorkerConfig,
    Field(discriminator="type"),
]
