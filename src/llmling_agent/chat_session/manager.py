"""Chat session management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from llmling_agent.chat_session.base import AgentChatSession
from llmling_agent.chat_session.exceptions import ChatSessionNotFoundError


if TYPE_CHECKING:
    from llmling_agent import LLMlingAgent
    from llmling_agent.delegation.pool import AgentPool


class ChatSessionManager:
    """Manages multiple agent chat sessions.

    Can be used as an async context manager to ensure proper cleanup:

    async with ChatSessionManager() as manager:
        session = await manager.create_session(...)
        # Use session...
    # All sessions and pools cleaned up automatically
    """

    def __init__(self):
        self._sessions: dict[str, AgentChatSession] = {}
        self._pools: dict[str, AgentPool] = {}  # Track pools per session

    async def create_session(
        self,
        agent: LLMlingAgent[Any, str],
        *,
        pool: AgentPool | None = None,
        wait_chain: bool = True,
    ) -> AgentChatSession:
        """Create and register a new session.

        Args:
            agent: The agent to create a session for
            pool: Optional agent pool for multi-agent interactions
            wait_chain: Whether to wait for chain completion

        Returns:
            New chat session instance
        """
        session = AgentChatSession(
            agent,
            pool=pool,
            wait_chain=wait_chain,
        )
        await session.initialize()
        self._sessions[session.id] = session
        if pool:
            self._pools[session.id] = pool
        return session

    def get_session(self, session_id: str) -> AgentChatSession:
        """Get an existing session.

        Args:
            session_id: ID of the session to retrieve

        Returns:
            The requested chat session

        Raises:
            ChatSessionNotFoundError: If session doesn't exist
        """
        try:
            return self._sessions[session_id]
        except KeyError as e:
            msg = f"Session {session_id} not found"
            raise ChatSessionNotFoundError(msg) from e

    async def end_session(self, session_id: str):
        """End and cleanup a session.

        Args:
            session_id: ID of the session to end
        """
        if session := self._sessions.pop(session_id, None):
            await session.cleanup()

        # Cleanup pool if it exists
        if pool := self._pools.pop(session_id, None):
            await pool.cleanup()

    async def cleanup(self):
        """Clean up all sessions and pools."""
        # Make copies since we'll modify during iteration
        sessions = list(self._sessions.keys())
        for session_id in sessions:
            await self.end_session(session_id)

        self._sessions.clear()
        self._pools.clear()

    def get_pool(self, session_id: str) -> AgentPool | None:
        """Get the agent pool associated with a session.

        Args:
            session_id: ID of the session

        Returns:
            Associated agent pool or None if no pool exists
        """
        return self._pools.get(session_id)

    async def __aenter__(self) -> Self:
        """Enter async context."""
        return self

    async def __aexit__(self, *exc: object):
        """Exit async context, cleaning up all sessions."""
        await self.cleanup()
