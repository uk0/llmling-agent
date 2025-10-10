from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = ["RpcTask", "RpcTaskKind"]


class RpcTaskKind(Enum):
    REQUEST = "request"
    NOTIFICATION = "notification"


@dataclass(slots=True)
class RpcTask:
    kind: RpcTaskKind
    message: dict[str, Any]


from .dispatcher import (
    DefaultMessageDispatcher,
    MessageDispatcher,
    NotificationRunner,
    RequestRunner,
)
from .queue import InMemoryMessageQueue, MessageQueue
from .sender import DebugCallback, MessageSender, SenderFactory
from .state import InMemoryMessageStateStore, MessageStateStore
from .supervisor import TaskSupervisor

__all__ += [
    "DebugCallback",
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
