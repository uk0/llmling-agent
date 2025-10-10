"""Task package."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = ["RpcTask", "RpcTaskKind"]


class RpcTaskKind(Enum):
    """RpcTaskKind represents the kind of task to be executed by the agent."""

    REQUEST = "request"
    NOTIFICATION = "notification"


@dataclass(slots=True)
class RpcTask:
    """RpcTask represents a task to be executed by the agent."""

    kind: RpcTaskKind
    message: dict[str, Any]


from .dispatcher import (
    DefaultMessageDispatcher,
    MessageDispatcher,
    NotificationRunner,
    RequestRunner,
)
from .queue import InMemoryMessageQueue, MessageQueue
from .sender import MessageSender, SenderFactory
from .state import InMemoryMessageStateStore, MessageStateStore
from .supervisor import TaskSupervisor
from .debug import DebugEntry, DebuggingMessageStateStore

__all__ += [
    "DebugEntry",
    "DebuggingMessageStateStore",
    "DefaultMessageDispatcher",
    "InMemoryMessageQueue",
    "InMemoryMessageStateStore",
    "MessageDispatcher",
    "MessageQueue",
    "MessageSender",
    "MessageStateStore",
    "NotificationRunner",
    "RequestRunner",
    "SenderFactory",
    "TaskSupervisor",
]
