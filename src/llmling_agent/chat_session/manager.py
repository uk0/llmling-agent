"""Chat session management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.chat_session.base import AgentChatSession
from llmling_agent.chat_session.exceptions import ChatSessionNotFoundError


if TYPE_CHECKING:
    from uuid import UUID

    from llmling_agent import LLMlingAgent


class ChatSessionManager:
    """Manages multiple agent chat sessions."""

    def __init__(self) -> None:
        self._sessions: dict[UUID, AgentChatSession] = {}

    async def create_session(
        self,
        agent: LLMlingAgent[str],
        *,
        model: str | None = None,
    ) -> AgentChatSession:
        """Create and register a new session."""
        model_override = model if model and model.strip() else None

        session = AgentChatSession(agent, model_override=model_override)
        await session.initialize()
        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: UUID) -> AgentChatSession:
        """Get an existing session."""
        try:
            return self._sessions[session_id]
        except KeyError as e:
            msg = f"Session {session_id} not found"
            raise ChatSessionNotFoundError(msg) from e

    def end_session(self, session_id: UUID) -> None:
        """End and cleanup a session."""
        self._sessions.pop(session_id, None)
