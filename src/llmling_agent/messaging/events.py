"""Event sources for LLMling agent."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import TYPE_CHECKING, Any, Literal, Self


ChangeType = Literal["added", "modified", "deleted"]


if TYPE_CHECKING:
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.models.events import ConnectionEventType, EventSourceConfig
    from llmling_agent.talk.talk import Talk


@dataclass(frozen=True, kw_only=True)
class EventData:
    """Base class for event data."""

    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, source: str, **kwargs: Any) -> Self:
        """Create event with current timestamp."""
        return cls(source=source, timestamp=datetime.now(), **kwargs)

    @abstractmethod
    def to_prompt(self) -> str:
        """Convert event to agent prompt."""

    async def format(self, config: EventSourceConfig) -> str:
        """Wraps core message with configurable template."""
        from jinja2 import Environment

        env = Environment(trim_blocks=True, lstrip_blocks=True)
        template = env.from_string(config.template)

        return template.render(
            source=self.source,
            content=self.to_prompt(),  # Use the core message
            metadata=self.metadata,
            timestamp=self.timestamp,
            include_metadata=config.include_metadata,
            include_timestamp=config.include_timestamp,
        )


@dataclass(frozen=True, kw_only=True)
class UIEvent(EventData):
    """Event triggered through UI interaction."""

    type: Literal["command", "message", "agent_command", "agent_message"]
    """Type of UI interaction that triggered this event."""

    content: str
    """The actual content (command string, voice command, etc.)."""

    args: list[str] = field(default_factory=list)
    """Additional arguments for the interaction."""

    kwargs: dict[str, Any] = field(default_factory=dict)
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


@dataclass(frozen=True)
class ConnectionEvent[TTransmittedData](EventData):
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


@dataclass(frozen=True, kw_only=True)
class FileEvent(EventData):
    """File system event."""

    path: str
    type: ChangeType

    def to_prompt(self) -> str:
        return f"File {self.type}: {self.path}"


@dataclass(frozen=True)
class FunctionResultEvent(EventData):
    """Event from a function execution result."""

    result: Any

    def to_prompt(self) -> str:
        """Convert result to prompt format."""
        return str(self.result)


@dataclass(frozen=True)
class EmailEvent(EventData):
    """Email event with specific content structure."""

    subject: str
    sender: str
    body: str

    def to_prompt(self) -> str:
        """Core email message."""
        return f"Email from {self.sender} with subject: {self.subject}\n\n{self.body}"


@dataclass(frozen=True)
class TimeEvent(EventData):
    """Time-based event."""

    schedule: str
    """Cron expression that triggered this event."""

    prompt: str
    """Cron expression that triggered this event."""

    def to_prompt(self) -> str:
        """Format scheduled event."""
        return f"Scheduled task triggered by {self.schedule}: {self.prompt}"


@dataclass(frozen=True)
class WebhookEvent(EventData):
    """Webhook payload with formatting."""

    payload: dict[str, Any]

    def to_prompt(self) -> str:
        """Format webhook payload."""
        return f"Webhook received:\n{json.dumps(self.payload, indent=2)}"
