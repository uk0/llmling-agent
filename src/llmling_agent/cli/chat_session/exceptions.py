from __future__ import annotations


class ChatError(Exception):
    """Base class for chat exceptions."""


class CommandError(ChatError):
    """Command execution error."""


class StateError(ChatError):
    """Session state error."""
