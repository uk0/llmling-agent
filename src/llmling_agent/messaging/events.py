"""Event sources for LLMling agent."""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime  # noqa: TC003
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from llmling_agent.messaging.messages import ChatMessage  # noqa: TC001
from llmling_agent.talk.talk import Talk  # noqa: TC001
from llmling_agent.utils.now import get_now
from llmling_agent_config.events import (  # noqa: TC001
    ConnectionEventType,
    EventSourceConfig,
)


ChangeType = Literal["added", "modified", "deleted"]


class EventData(BaseModel):
    """Base class for event data."""

    source: str
    timestamp: datetime = Field(default_factory=get_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        frozen=True,
        use_attribute_docstrings=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @classmethod
    def create(cls, source: str, **kwargs: Any) -> Self:
        """Create event with current timestamp."""
        return cls(source=source, **kwargs)

    @abstractmethod
    def to_prompt(self) -> str:
        """Convert event to agent prompt."""

    async def format(self, config: EventSourceConfig) -> str:
        """Wraps core message with configurable template."""
        from jinjarope import Environment

        env = Environment(trim_blocks=True, lstrip_blocks=True, enable_async=True)
        template = env.from_string(config.template)

        return await template.render_async(
            source=self.source,
            content=self.to_prompt(),  # Use the core message
            metadata=self.metadata,
            timestamp=self.timestamp,
            include_metadata=config.include_metadata,
            include_timestamp=config.include_timestamp,
        )


class UIEventData(EventData):
    """Event triggered through UI interaction."""

    type: Literal["command", "message", "agent_command", "agent_message"]
    """Type of UI interaction that triggered this event."""

    content: str
    """The actual content (command string, voice command, etc.)."""

    args: list[str] = Field(default_factory=list)
    """Additional arguments for the interaction."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Additional options/parameters."""

    agent_name: str | None = None
    """Target agent for @agent messages/commands."""

    def to_prompt(self) -> str:
        """Convert event to agent prompt."""
        match self.type:
            case "command":
                args_str = " ".join(self.args)
                kwargs_str = " ".join(f"--{k}={v}" for k, v in self.kwargs.items())
                return f"UI Command: /{self.content} {args_str} {kwargs_str}"
            case "shortcut" | "gesture":
                return f"UI Action: {self.content}"
            case "voice":
                return f"Voice Command: {self.content}"
            case _:
                raise ValueError(self.type)


class ConnectionEventData[TTransmittedData](EventData):
    """Event from connection activity."""

    connection_name: str
    """Name of the connection which fired an event."""

    connection: Talk[TTransmittedData]
    """The connection which fired the event."""

    event_type: ConnectionEventType
    """Type of event that occurred."""

    message: ChatMessage[TTransmittedData] | None = None
    """The message at the stage of the event."""

    def to_prompt(self) -> str:
        """Convert event to agent prompt."""
        base = f"Connection '{self.connection_name}' event: {self.event_type}"
        if self.message:
            return f"{base}\nMessage content: {self.message.content}"
        return base


class FileEventData(EventData):
    """File system event."""

    path: str
    type: ChangeType

    def to_prompt(self) -> str:
        return f"File {self.type}: {self.path}"


class FunctionResultEventData(EventData):
    """Event from a function execution result."""

    result: Any

    def to_prompt(self) -> str:
        """Convert result to prompt format."""
        return str(self.result)


class EmailEventData(EventData):
    """Email event with specific content structure."""

    subject: str
    sender: str
    body: str

    def to_prompt(self) -> str:
        """Core email message."""
        return f"Email from {self.sender} with subject: {self.subject}\n\n{self.body}"


class TimeEventData(EventData):
    """Time-based event."""

    schedule: str
    """Cron expression that triggered this event."""

    prompt: str
    """Cron expression that triggered this event."""

    def to_prompt(self) -> str:
        """Format scheduled event."""
        return f"Scheduled task triggered by {self.schedule}: {self.prompt}"


class WebhookEventData(EventData):
    """Webhook payload with formatting."""

    payload: dict[str, Any]

    def to_prompt(self) -> str:
        """Format webhook payload."""
        import anyenv

        return f"Webhook received:\n{anyenv.dump_json(self.payload, indent=True)}"
