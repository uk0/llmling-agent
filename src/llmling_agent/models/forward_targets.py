"""Forward target models."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from upath import UPath


ConnectionType = Literal["run", "context", "forward"]


class ConnectionConfig(BaseModel):
    """Base model for message forwarding targets."""

    type: str = Field(init=False)
    """Discriminator field for forward target types."""

    wait_for_completion: bool = Field(True)
    """Whether to wait for the result before continuing.

    If True, message processing will wait for the target to complete.
    If False, message will be forwarded asynchronously.
    """
    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class AgentConnectionConfig(ConnectionConfig):
    """Forward messages to another agent.

    This configuration defines how messages should flow from one agent to another,
    including:
    - Basic routing (which agent, what type of connection)
    - Message queueing and processing strategies
    - Timing controls (priority, delay)
    - Execution behavior (wait for completion)

    Example:
        ```yaml
        agents:
          analyzer:
            forward_to:
              - type: agent
                name: planner
                connection_type: run
                queued: true
                queue_strategy: concat
                priority: 1
                delay: 5s
                wait_for_completion: true
        ```
    """

    type: Literal["agent"] = Field("agent", init=False)
    """Type discriminator for agent targets."""

    name: str
    """Name of target agent."""

    connection_type: ConnectionType = "run"
    """How messages should be handled by the target agent:
    - run: Execute message as a new run
    - context: Add message to agent's context
    - forward: Forward message to agent's outbox
    """

    queued: bool = False
    """Whether messages should be queued for manual processing."""

    queue_strategy: Literal["concat", "latest", "buffer"] = "latest"
    """How to process queued messages:
    - concat: Combine all messages with newlines
    - latest: Use only the most recent message
    - buffer: Process all messages individually
    """

    priority: int = 0
    """Priority of the task. Lower = higher priority."""

    delay: timedelta | None = None
    """Delay before running the task."""

    filter: str | None = None
    """Optional filter condition for message forwarding.
    Can be a string containing a Python expression that will be evaluated
    with the message available as 'message'.
    """


class FileConnectionConfig(ConnectionConfig):
    """Save messages to a file."""

    type: Literal["file"] = Field("file", init=False)

    path: str
    """Path to output file. Supports variables: {date}, {time}, {agent}"""

    def resolve_path(self, context: dict[str, str]) -> UPath:
        """Resolve path template with context variables."""
        now = datetime.now()
        variables = {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H-%M-%S"),
            **context,
        }
        return UPath(self.path.format(**variables))


ForwardingTarget = Annotated[
    AgentConnectionConfig | FileConnectionConfig, Field(discriminator="type")
]
