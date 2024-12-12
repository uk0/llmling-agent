"""Session-specific exceptions."""

from __future__ import annotations


class ChatSessionError(Exception):
    """Base exception for chat session-related errors."""


class ChatSessionNotFoundError(ChatSessionError):
    """Raised when trying to access a non-existent session."""


class ChatSessionConfigError(ChatSessionError):
    """Raised when chat session configuration is invalid."""
