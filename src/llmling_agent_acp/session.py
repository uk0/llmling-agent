"""ACP (Agent Client Protocol) session management for llmling-agent.

This module provides session lifecycle management, state tracking, and coordination
between agents and ACP clients through the JSON-RPC protocol.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Self
import uuid

from llmling_agent.log import get_logger
from llmling_agent_acp.converters import (
    FileSystemBridge,
    format_tool_call_for_acp,
    from_content_blocks,
    to_session_updates,
)


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from acp import Client, SessionNotification
    from acp.schema import McpServer

    from llmling_agent import Agent
    from llmling_agent_acp.types import ContentBlock

logger = get_logger(__name__)


class ACPSession:
    """Individual ACP session state and management.

    Manages the lifecycle and state of a single ACP session, including:
    - Agent instance and conversation state
    - Working directory and environment
    - MCP server connections
    - File system bridge for client operations
    - Tool execution and streaming updates
    """

    def __init__(
        self,
        session_id: str,
        agent: Agent[Any],
        cwd: str,
        client: Client,
        mcp_servers: list[McpServer] | None = None,
    ) -> None:
        """Initialize ACP session.

        Args:
            session_id: Unique session identifier
            agent: llmling agent instance for this session
            cwd: Working directory for the session
            client: External library Client interface for operations
            mcp_servers: Optional MCP server configurations
        """
        self.session_id = session_id
        self.agent = agent
        self.cwd = cwd
        self.client = client
        self.mcp_servers = mcp_servers or []

        # Session state
        self._active = True
        self._conversation_history: list[dict[str, Any]] = []
        self._task_lock = asyncio.Lock()

        # File system bridge
        self.fs_bridge = FileSystemBridge()

        logger.info("Created ACP session %s with agent %s", session_id, agent.name)

    @property
    def active(self) -> bool:
        """Check if session is active."""
        return self._active

    async def process_prompt(
        self, content_blocks: list[ContentBlock]
    ) -> AsyncGenerator[SessionNotification, None]:
        """Process a prompt request and stream responses.

        Args:
            content_blocks: List of content blocks from the prompt request

        Yields:
            SessionNotification objects for streaming to client
        """
        if not self._active:
            logger.warning(
                "Attempted to process prompt on inactive session %s", self.session_id
            )
            return

        async with self._task_lock:
            try:
                # Convert content blocks to prompt text
                prompt_text = from_content_blocks(content_blocks)

                if not prompt_text.strip():
                    logger.warning(
                        "Empty prompt received for session %s", self.session_id
                    )
                    return

                logger.debug(
                    "Processing prompt for session %s: %s",
                    self.session_id,
                    prompt_text[:100],
                )

                # Store user message in conversation history
                self._conversation_history.append({
                    "role": "user",
                    "content": prompt_text,
                    "content_blocks": content_blocks,
                })

                # Check if agent supports streaming
                if hasattr(self.agent, "run_stream"):
                    async for notification in self._process_streaming_response(
                        prompt_text
                    ):
                        yield notification
                else:
                    async for notification in self._process_sync_response(prompt_text):
                        yield notification

            except Exception as e:
                logger.exception("Error processing prompt in session %s", self.session_id)
                # Send error as agent message
                error_updates = to_session_updates(
                    f"I encountered an error while processing your request: {e}",
                    self.session_id,
                )
                for update in error_updates:
                    yield update

    async def _process_streaming_response(
        self, prompt: str
    ) -> AsyncGenerator[SessionNotification, None]:
        """Process prompt with streaming response.

        Args:
            prompt: Prompt text to process

        Yields:
            SessionNotification objects
        """
        try:
            response_parts = []

            async with self.agent.run_stream(prompt) as stream:
                async for chunk in stream.stream_text(delta=True):
                    if chunk and str(chunk).strip():
                        chunk_text = str(chunk)
                        response_parts.append(chunk_text)

                        # Stream individual chunks
                        chunk_updates = to_session_updates(chunk_text, self.session_id)
                        for update in chunk_updates:
                            yield update

            # Process complete response after streaming
            if response_parts:
                complete_response = "".join(response_parts)

                # Store complete response in history
                self._conversation_history.append({
                    "role": "assistant",
                    "content": complete_response,
                })

        except Exception as e:
            logger.exception(
                "Error in streaming response for session %s", self.session_id
            )
            error_updates = to_session_updates(f"Streaming error: {e}", self.session_id)
            for update in error_updates:
                yield update

    async def _process_sync_response(
        self, prompt: str
    ) -> AsyncGenerator[SessionNotification, None]:
        """Process prompt with synchronous response.

        Args:
            prompt: Prompt text to process

        Yields:
            SessionNotification objects
        """
        try:
            # Execute agent synchronously
            result = await self.agent.run(prompt)
            response_text = str(result.content) if result.content is not None else ""

            if response_text.strip():
                # Store response in history
                self._conversation_history.append({
                    "role": "assistant",
                    "content": response_text,
                })

                # Convert to session updates for streaming
                updates = to_session_updates(response_text, self.session_id)
                for update in updates:
                    yield update

        except Exception as e:
            logger.exception("Error in sync response for session %s", self.session_id)
            error_updates = to_session_updates(f"Processing error: {e}", self.session_id)
            for update in error_updates:
                yield update

    async def execute_tool(
        self, tool_name: str, tool_params: dict[str, Any]
    ) -> AsyncGenerator[SessionNotification, None]:
        """Execute a tool and stream the results.

        Args:
            tool_name: Name of the tool to execute
            tool_params: Parameters to pass to the tool

        Yields:
            SessionNotification objects for tool execution updates
        """
        try:
            # Get the tool using ToolManager's dict-like access
            try:
                tool = self.agent.tools[tool_name]
            except KeyError:
                logger.warning(
                    "Tool %s not found in agent %s", tool_name, self.agent.name
                )
                return

            # Execute the tool using Tool.execute() method
            result = await tool.execute(**tool_params)

            # Format as ACP tool call notification
            notification = format_tool_call_for_acp(
                tool_name=tool_name,
                tool_input=tool_params,
                tool_output=result,
                session_id=self.session_id,
                status="completed",
            )

            yield notification

        except Exception as e:
            logger.exception(
                "Error executing tool %s in session %s", tool_name, self.session_id
            )

            # Send error notification
            error_notification = format_tool_call_for_acp(
                tool_name=tool_name,
                tool_input=tool_params,
                tool_output=f"Error: {e}",
                session_id=self.session_id,
                status="error",
            )

            yield error_notification

    async def load_conversation_history(self, history: list[dict[str, Any]]) -> None:
        """Load conversation history into the session.

        Args:
            history: List of conversation messages
        """
        try:
            self._conversation_history = history.copy()

            # Convert to ChatMessage format for agent
            from llmling_agent.messaging.messages import ChatMessage

            chat_messages = []
            for msg in history:
                chat_msg = ChatMessage[str](
                    content=msg.get("content", ""),
                    role=msg.get("role", "user"),
                    name=self.agent.name if msg.get("role") == "assistant" else "user",
                )
                chat_messages.append(chat_msg)

            # Set conversation history in agent
            if hasattr(self.agent, "conversation") and hasattr(
                self.agent.conversation, "set_history"
            ):
                self.agent.conversation.set_history(chat_messages)

            logger.info(
                "Loaded %d messages into session %s history",
                len(history),
                self.session_id,
            )

        except Exception:
            logger.exception(
                "Error loading conversation history for session %s", self.session_id
            )

    async def get_conversation_history(self) -> list[dict[str, Any]]:
        """Get the conversation history for this session.

        Returns:
            List of conversation messages
        """
        return self._conversation_history.copy()

    async def close(self) -> None:
        """Close the session and cleanup resources."""
        if not self._active:
            return

        self._active = False

        try:
            # Cleanup agent resources
            if hasattr(self.agent, "__aexit__"):
                await self.agent.__aexit__(None, None, None)

            logger.info("Closed ACP session %s", self.session_id)

        except Exception:
            logger.exception("Error closing session %s", self.session_id)


class ACPSessionManager:
    """Manages multiple ACP sessions and their lifecycle.

    Provides centralized management of ACP sessions, including:
    - Session creation and initialization
    - Session lookup and retrieval
    - Session cleanup and resource management
    - Agent instance management
    """

    def __init__(self) -> None:
        """Initialize session manager."""
        self._sessions: dict[str, ACPSession] = {}
        self._lock = asyncio.Lock()

        logger.info("Initialized ACP session manager")

    async def create_session(
        self,
        agent: Agent[Any],
        cwd: str,
        client: Client,
        mcp_servers: list[McpServer] | None = None,
        session_id: str | None = None,
    ) -> str:
        """Create a new ACP session.

        Args:
            agent: llmling agent instance for the session
            cwd: Working directory for the session
            client: External library Client interface
            mcp_servers: Optional MCP server configurations
            session_id: Optional specific session ID (generated if None)

        Returns:
            Session ID for the created session
        """
        async with self._lock:
            # Generate session ID if not provided
            if session_id is None:
                session_id = f"sess_{uuid.uuid4().hex[:12]}"

            # Check for existing session
            if session_id in self._sessions:
                logger.warning("Session ID %s already exists", session_id)
                msg = f"Session {session_id} already exists"
                raise ValueError(msg)

            # Create session
            session = ACPSession(
                session_id=session_id,
                agent=agent,
                cwd=cwd,
                client=client,
                mcp_servers=mcp_servers,
            )

            # Store session
            self._sessions[session_id] = session

            logger.info("Created session %s with agent %s", session_id, agent.name)
            return session_id

    async def get_session(self, session_id: str) -> ACPSession | None:
        """Get an existing session by ID.

        Args:
            session_id: Session identifier

        Returns:
            ACPSession instance or None if not found
        """
        return self._sessions.get(session_id)

    async def close_session(self, session_id: str) -> None:
        """Close and remove a session.

        Args:
            session_id: Session identifier to close
        """
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                await session.close()
                logger.info("Removed session %s", session_id)
            else:
                logger.warning("Attempted to close non-existent session %s", session_id)

    async def list_sessions(self) -> list[str]:
        """List all active session IDs.

        Returns:
            List of active session IDs
        """
        return list(self._sessions.keys())

    async def get_session_count(self) -> int:
        """Get the number of active sessions.

        Returns:
            Number of active sessions
        """
        return len(self._sessions)

    async def close_all_sessions(self) -> None:
        """Close all active sessions."""
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        # Close sessions outside of lock to avoid deadlock
        for session in sessions:
            try:
                await session.close()
            except Exception:
                logger.exception("Error closing session %s", session.session_id)

        logger.info("Closed all %d sessions", len(sessions))

    async def cleanup_inactive_sessions(self) -> None:
        """Remove any inactive sessions."""
        async with self._lock:
            inactive_sessions = [
                session_id
                for session_id, session in self._sessions.items()
                if not session.active
            ]

            for session_id in inactive_sessions:
                session = self._sessions.pop(session_id, None)
                if session:
                    try:
                        await session.close()
                    except Exception:
                        logger.exception("Error closing inactive session %s", session_id)

            if inactive_sessions:
                logger.info("Cleaned up %d inactive sessions", len(inactive_sessions))

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close_all_sessions()
