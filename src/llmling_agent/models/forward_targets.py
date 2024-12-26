"""Forward target models."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from upath import UPath


if TYPE_CHECKING:
    from llmling_agent.agent.agent import LLMlingAgent
    from llmling_agent.models.messages import ChatMessage


class ForwardTarget(BaseModel):
    """Base model for message forwarding targets."""

    type: str = Field(init=False)
    """Discriminator field for forward target types."""

    wait_for_completion: bool = Field(True)
    """Whether to wait for the result before continuing.

    If True, message processing will wait for the target to complete.
    If False, message will be forwarded asynchronously.
    """
    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class AgentTarget(ForwardTarget):
    """Forward messages to another agent."""

    type: Literal["agent"] = Field("agent", init=False)
    """Type discriminator for agent targets."""

    name: str
    """Name of target agent."""


class FileTarget(ForwardTarget):
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


ForwardingTarget = Annotated[AgentTarget | FileTarget, Field(discriminator="type")]


async def write_output(target: FileTarget, content: str, context: dict[str, str]):
    """Write content to file target."""
    path = target.resolve_path(context)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content))


# Example usage in LLMlingAgent._handle_message:
async def _handle_message(
    self, source: LLMlingAgent[Any, Any], message: ChatMessage[Any]
):
    """Handle a message forwarded from another agent."""
    context = {"agent": self.name}

    for target in self._context.config.forward_to:
        match target:
            case AgentTarget():
                # Create task for agent forwarding
                loop = asyncio.get_event_loop()
                task = loop.create_task(self.run(str(message.content)))
                self._pending_tasks.add(task)
                task.add_done_callback(self._pending_tasks.discard)

            case FileTarget():
                await write_output(target, str(message.content), context)
