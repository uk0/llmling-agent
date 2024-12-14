"""Runner exceptions."""

from __future__ import annotations


class AgentOrchestratorError(Exception):
    """Base exception for orchestrator errors."""


class AgentNotFoundError(AgentOrchestratorError):
    """Raised when requested agent is not found."""


class NoPromptsError(AgentOrchestratorError):
    """Raised when no prompts are provided."""


class AgentRunnerError(Exception):
    """Base exception for agent runner errors."""


class NotInitializedError(AgentRunnerError):
    """Raised when trying to access runner not initialized via context manager."""
