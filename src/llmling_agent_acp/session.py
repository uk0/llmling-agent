"""ACP (Agent Client Protocol) session management for llmling-agent.

This module provides session lifecycle management, state tracking, and coordination
between agents and ACP clients through the JSON-RPC protocol.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Self
import uuid

from pydantic_ai import Agent as PydanticAIAgent, ModelRequestNode

from llmling_agent.log import get_logger
from llmling_agent.mcp_server.manager import MCPManager
from llmling_agent_acp.converters import (
    FileSystemBridge,
    convert_acp_mcp_server_to_config,
    create_thought_chunk,
    format_tool_call_for_acp,
    from_content_blocks,
    to_session_updates,
)


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from acp import Client
    from acp.schema import McpServer
    from pydantic_ai.agent import CallToolsNode

    from llmling_agent import Agent
    from llmling_agent_acp.acp_types import ContentBlock, StopReason
    from llmling_agent_acp.command_bridge import ACPCommandBridge
    from llmling_agent_providers.base import AgentRunProtocol

from acp.schema import (
    AvailableCommand,
    SessionNotification,
    SessionUpdate7 as AvailableCommandsUpdate,
)


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
        max_turn_requests: int = 50,
        max_tokens: int | None = None,
        command_bridge: ACPCommandBridge | None = None,
    ) -> None:
        """Initialize ACP session.

        Args:
            session_id: Unique session identifier
            agent: llmling agent instance for this session
            cwd: Working directory for the session
            client: External library Client interface for operations
            mcp_servers: Optional MCP server configurations
            max_turn_requests: Maximum model requests per turn
            max_tokens: Maximum tokens per turn (if None, no limit)
            command_bridge: Optional command bridge for slash commands
        """
        self.session_id = session_id
        self.agent = agent
        self.cwd = cwd
        self.client = client
        self.mcp_servers = mcp_servers or []
        self.max_turn_requests = max_turn_requests
        self.max_tokens = max_tokens

        # Session state
        self._active = True
        self._task_lock = asyncio.Lock()
        self._cancelled = False
        self._current_turn_requests = 0
        self._current_turn_tokens = 0

        self.fs_bridge = FileSystemBridge()
        self.command_bridge = command_bridge

        # MCP integration
        self.mcp_manager: MCPManager | None = None

        logger.info("Created ACP session %s with agent %s", session_id, agent.name)

    async def initialize_mcp_servers(self) -> None:
        """Initialize MCP servers if any are configured."""
        if not self.mcp_servers:
            return

        logger.info(
            "Initializing %d MCP servers for session %s",
            len(self.mcp_servers),
            self.session_id,
        )

        try:
            # Convert ACP McpServer configs to our format
            server_configs = [
                convert_acp_mcp_server_to_config(server) for server in self.mcp_servers
            ]

            # Initialize MCP manager with converted configs
            self.mcp_manager = MCPManager(
                name=f"session_{self.session_id}",
                servers=server_configs,
                context=self.agent.context,
            )

            # Start MCP manager and get tools
            await self.mcp_manager.__aenter__()

            # Add MCP tools to the agent's tool system
            mcp_tools = await self.mcp_manager.get_tools()

            # Register MCP tools with the agent
            for tool in mcp_tools:
                self.agent.tools.register_tool(tool)

            logger.info(
                "Added %d MCP tools to agent for session %s",
                len(mcp_tools),
                self.session_id,
            )

            # Update available commands since new tools may affect command context
            await self.send_available_commands_update()

        except Exception:
            logger.exception(
                "Failed to initialize MCP servers for session %s", self.session_id
            )
            # Don't fail session creation, just log the error
            self.mcp_manager = None

    @property
    def active(self) -> bool:
        """Check if session is active."""
        return self._active

    def cancel(self) -> None:
        """Cancel the current prompt turn."""
        self._cancelled = True
        logger.info("Session %s cancelled", self.session_id)

    def is_cancelled(self) -> bool:
        """Check if the session is cancelled."""
        return self._cancelled

    async def process_prompt(
        self, content_blocks: list[ContentBlock]
    ) -> AsyncGenerator[SessionNotification | StopReason, None]:
        """Process a prompt request and stream responses.

        Args:
            content_blocks: List of content blocks from the prompt request

        Yields:
            SessionNotification objects for streaming to client, or StopReason literal
        """
        if not self._active:
            msg = "Attempted to process prompt on inactive session %s"
            logger.warning(msg, self.session_id)
            yield "refusal"
            return

        # Reset turn counters
        self._cancelled = False
        self._current_turn_requests = 0
        self._current_turn_tokens = 0

        async with self._task_lock:
            try:
                # Check for cancellation
                if self._cancelled:
                    yield "cancelled"
                    return

                # Convert content blocks to prompt text
                prompt_text = from_content_blocks(content_blocks)
                blocks = [
                    {"type": b.type, "content": str(b)[:50]} for b in content_blocks
                ]
                logger.info("Content blocks received: %s", blocks)
                logger.info("Converted prompt text: %r", prompt_text)

                if not prompt_text.strip():
                    msg = "Empty prompt received for session %s"
                    logger.warning(msg, self.session_id)
                    yield "refusal"
                    return

                # Check for slash commands
                if self.command_bridge and self.command_bridge.is_slash_command(
                    prompt_text
                ):
                    logger.info("Processing slash command: %s", prompt_text)
                    async for notification in self.command_bridge.execute_slash_command(
                        prompt_text, self
                    ):
                        yield notification
                    yield "end_turn"
                    return
                msg = "Processing prompt for session %s: %s"
                logger.debug(msg, self.session_id, prompt_text[:100])

                # Use iterate_run for comprehensive streaming
                msg = "Starting _process_iter_response for session %s"
                logger.info(msg, self.session_id)
                notification_count = 0
                stop_reason = None
                async for result in self._process_iter_response(prompt_text):
                    if isinstance(result, str):
                        # Stop reason received
                        stop_reason = result
                        break
                    else:
                        # Session notification
                        notification_count += 1
                        msg = "Yielding notification %d for session %s"
                        logger.info(msg, notification_count, self.session_id)
                        yield result

                # Yield the final stop reason
                final_stop_reason = stop_reason or "end_turn"
                msg = "Finished streaming, sent %d notifications, stop reason: %s"
                logger.info(msg, notification_count, final_stop_reason)
                yield final_stop_reason

            except Exception as e:
                logger.exception("Error processing prompt in session %s", self.session_id)
                # Send error as agent message
                msg = f"I encountered an error while processing your request: {e}"
                error_updates = to_session_updates(msg, self.session_id)
                for update in error_updates:
                    yield update
                # Return refusal for errors
                yield "refusal"

    async def _process_iter_response(  # noqa: PLR0915
        self, prompt: str
    ) -> AsyncGenerator[SessionNotification | StopReason, None]:
        """Process prompt using agent iteration for comprehensive streaming.

        Args:
            prompt: Prompt text to process

        Yields:
            SessionNotification objects for all agent execution events,
            or StopReason literal
        """
        from acp.schema import ContentBlock1, SessionUpdate2

        try:
            response_parts = []
            msg = "Starting agent.iterate_run for session %s with prompt: %r"
            logger.info(msg, self.session_id, prompt[:100])
            logger.info("Agent model: %s", getattr(self.agent, "model_name", "unknown"))

            async with self.agent.iterate_run(prompt) as agent_run:
                logger.info("Agent run started for session %s", self.session_id)
                node_count = 0
                has_yielded_anything = False
                async for node in agent_run:
                    # Check for cancellation
                    if self._cancelled:
                        yield "cancelled"
                        return

                    # Check turn limits
                    if self._current_turn_requests >= self.max_turn_requests:
                        yield "max_turn_requests"
                        return

                    node_count += 1
                    msg = "Processing node %d (%s) for session %s"
                    logger.info(msg, node_count, type(node).__name__, self.session_id)
                    if PydanticAIAgent.is_user_prompt_node(node):
                        # User prompt node - log but don't stream (already processed)
                        msg = "Processing user prompt node for session %s"
                        logger.debug(msg, self.session_id)

                    elif PydanticAIAgent.is_model_request_node(node):
                        # Increment request counter
                        self._current_turn_requests += 1

                        # Model request node - stream the model's response
                        msg = "Starting model request streaming for session %s"
                        logger.info(msg, self.session_id)
                        notification_count = 0
                        async for result in self._stream_model_request(node, agent_run):
                            if isinstance(result, str):
                                # Stop reason from model request
                                yield result
                                return
                            elif result:
                                notification_count += 1
                                has_yielded_anything = True
                                msg = "Yielding model notification %d for session %s"
                                logger.info(msg, notification_count, self.session_id)
                                yield result
                        msg = "Model request streaming finished, yielded %d notifications"
                        logger.info(msg, notification_count)

                    elif PydanticAIAgent.is_call_tools_node(node):
                        # Tool execution node - stream tool calls and results
                        msg = "Starting tool execution streaming for session %s"
                        logger.info(msg, self.session_id)
                        async for notification in self._stream_tool_execution(
                            node, agent_run
                        ):
                            if notification:
                                has_yielded_anything = True
                                yield notification

                    elif (
                        PydanticAIAgent.is_end_node(node)
                        and agent_run.result
                        and agent_run.result.output
                    ):
                        final_content = str(agent_run.result.output)
                        msg = "End node reached with output: %r"
                        logger.info(msg, final_content[:100])
                        if final_content.strip():
                            response_parts.append(final_content)

                            # Send final response as session update if nothing streamed
                            if not has_yielded_anything:
                                msg = (
                                    "No streaming occurred,"
                                    "sending final response for session %s"
                                )
                                logger.info(msg, self.session_id)
                                content_block = ContentBlock1(
                                    text=final_content, type="text"
                                )
                                update = SessionUpdate2(
                                    content=content_block,
                                    sessionUpdate="agent_message_chunk",
                                )
                                notification = SessionNotification(
                                    sessionId=self.session_id, update=update
                                )
                                has_yielded_anything = True
                                yield notification
                            msg = "Agent iteration completed for session %s"
                            logger.debug(msg, self.session_id)
                    else:
                        msg = "Unknown node type for session %s: %s"
                        logger.info(msg, self.session_id, type(node).__name__)
                msg = "Agent iteration finished. Processed %d nodes, yielded anything: %s"
                logger.info(msg, node_count, has_yielded_anything)

        except Exception as e:
            logger.exception("Error in agent iteration for session %s", self.session_id)
            logger.info("Sending error updates for session %s", self.session_id)
            error_updates = to_session_updates(f"Agent error: {e}", self.session_id)
            for update in error_updates:
                yield update

    async def _stream_model_request(  # noqa: PLR0915
        self,
        node: ModelRequestNode,
        agent_run: AgentRunProtocol,
    ) -> AsyncGenerator[SessionNotification | StopReason, None]:
        """Stream model request events.

        Args:
            node: Model request node
            agent_run: Agent run context

        Yields:
            SessionNotification objects for model streaming, or StopReason literal
        """
        from acp.schema import ContentBlock1, SessionUpdate2
        from pydantic_ai.messages import (
            FinalResultEvent,
            PartDeltaEvent,
            PartStartEvent,
            TextPartDelta,
            ThinkingPartDelta,
            ToolCallPartDelta,
        )

        try:
            async with node.stream(agent_run.ctx) as request_stream:
                text_content = []
                event_count = 0
                msg = "Starting to iterate over request_stream events for session %s"
                logger.info(msg, self.session_id)
                async for event in request_stream:
                    # Check for cancellation
                    if self._cancelled:
                        yield "cancelled"
                        return

                    event_count += 1
                    msg = "Received event %d: %s for session %s"
                    logger.info(msg, event_count, type(event).__name__, self.session_id)
                    match event:
                        case PartStartEvent():
                            # Part started - could log but don't stream yet
                            pass

                        case PartDeltaEvent(
                            delta=TextPartDelta(content_delta=content)
                        ) if content:
                            msg = "Processing TextPartDelta %r for session %s"
                            logger.info(msg, content, self.session_id)
                            # Track tokens if limits are set
                            if self.max_tokens is not None:
                                # Rough token estimation (1 token ≈ 4 characters)
                                estimated_tokens = len(content) // 4
                                self._current_turn_tokens += estimated_tokens
                                if self._current_turn_tokens >= self.max_tokens:
                                    yield "max_tokens"
                                    return

                            # Stream text deltas directly as single agent message chunks
                            text_content.append(content)

                            # Create single chunk update directly
                            content_block = ContentBlock1(
                                text=content, type="text", annotations=None
                            )
                            update = SessionUpdate2(
                                content=content_block,
                                sessionUpdate="agent_message_chunk",
                            )
                            notification = SessionNotification(
                                sessionId=self.session_id, update=update
                            )
                            msg = "Yielding TextPartDelta notification for session %s"
                            logger.info(msg, self.session_id)
                            yield notification

                        case PartDeltaEvent(delta=TextPartDelta(content_delta=content)):
                            msg = (
                                "Received TextPartDelta with empty/falsy content:"
                                " %r for session %s"
                            )
                            logger.info(msg, content, self.session_id)

                        case PartDeltaEvent(
                            delta=ThinkingPartDelta(content_delta=content)
                        ) if content:
                            msg = "Processing ThinkingPartDelta %r for session %s"
                            logger.info(msg, content, self.session_id)
                            # Stream thinking as agent thought chunks
                            thought_notification = create_thought_chunk(
                                content, self.session_id
                            )
                            yield thought_notification

                        case PartDeltaEvent(delta=ToolCallPartDelta()):
                            msg = "Received ToolCallPartDelta for session %s"
                            logger.info(msg, self.session_id)
                        case FinalResultEvent():
                            msg = "Received FinalResultEvent for session %s"
                            logger.info(msg, self.session_id)
                            break

                        case _:
                            msg = "Received unhandled event type: %s for session %s"
                            logger.info(msg, type(event).__name__, self.session_id)

                # Log accumulated text content if any
                if text_content:
                    accumulated_text = "".join(text_content)
                    msg = "Received streaming response for session %s: %r"
                    logger.debug(msg, self.session_id, accumulated_text[:100])

        except Exception:
            msg = "Error streaming model request for session %s"
            logger.exception(msg, self.session_id)
            yield "refusal"

    async def _stream_tool_execution(
        self,
        node: CallToolsNode,
        agent_run: AgentRunProtocol,
    ) -> AsyncGenerator[SessionNotification, None]:
        """Stream tool execution events.

        Args:
            node: Tool execution node
            agent_run: Agent run context

        Yields:
            SessionNotification objects for tool execution
        """
        from pydantic_ai.messages import (
            FunctionToolCallEvent,
            FunctionToolResultEvent,
            RetryPromptPart,
            ToolReturnPart,
        )

        # Track tool call inputs by tool_call_id
        inputs: dict[str, dict] = {}

        try:
            async with node.stream(agent_run.ctx) as tool_stream:
                async for event in tool_stream:
                    match event:
                        case FunctionToolCallEvent() as tool_event:
                            # Tool call started - save input for later use
                            tool_call_id = tool_event.part.tool_call_id
                            inputs[tool_call_id] = tool_event.part.args_as_dict()

                            tool_notification = format_tool_call_for_acp(
                                tool_name=tool_event.part.tool_name,
                                tool_input=tool_event.part.args_as_dict(),
                                tool_output=None,  # Not available yet
                                session_id=self.session_id,
                                status="running",
                            )
                            yield tool_notification

                        case FunctionToolResultEvent() as result_event if isinstance(
                            result_event.result, ToolReturnPart
                        ):
                            # Tool call completed successfully
                            tool_call_id = result_event.tool_call_id
                            tool_input = inputs.get(tool_call_id, {})

                            tool_notification = format_tool_call_for_acp(
                                tool_name=result_event.result.tool_name,
                                tool_input=tool_input,
                                tool_output=result_event.result.content,
                                session_id=self.session_id,
                                status="completed",
                            )
                            yield tool_notification

                            # Clean up stored input
                            inputs.pop(tool_call_id, None)

                        case FunctionToolResultEvent() as result_event if isinstance(
                            result_event.result, RetryPromptPart
                        ):
                            # Tool call failed and needs retry
                            tool_call_id = result_event.tool_call_id
                            tool_input = inputs.get(tool_call_id, {})
                            tool_name = result_event.result.tool_name or "unknown"
                            error_message = result_event.result.model_response()

                            tool_notification = format_tool_call_for_acp(
                                tool_name=tool_name,
                                tool_input=tool_input,
                                tool_output=f"Error: {error_message}",
                                session_id=self.session_id,
                                status="failed",
                            )
                            yield tool_notification

                            # Clean up stored input
                            inputs.pop(tool_call_id, None)

        except Exception:
            msg = "Error streaming tool execution for session %s"
            logger.exception(msg, self.session_id)

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
            try:
                tool = self.agent.tools[tool_name]
            except KeyError:
                msg = "Tool %s not found in agent %s"
                logger.warning(msg, tool_name, self.agent.name)
                return
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
            msg = "Error executing tool %s in session %s"
            logger.exception(msg, tool_name, self.session_id)
            error_notification = format_tool_call_for_acp(
                tool_name=tool_name,
                tool_input=tool_params,
                tool_output=f"Error: {e}",
                session_id=self.session_id,
                status="error",
            )

            yield error_notification

    async def close(self) -> None:
        """Close the session and cleanup resources."""
        if not self._active:
            return

        self._active = False

        try:
            # Clean up MCP manager if present
            if self.mcp_manager:
                await self.mcp_manager.cleanup()
                self.mcp_manager = None

            await self.agent.__aexit__(None, None, None)
            logger.info("Closed ACP session %s", self.session_id)
        except Exception:
            logger.exception("Error closing session %s", self.session_id)

    async def send_available_commands_update(self) -> None:
        """Send current available commands to client."""
        if not self.command_bridge:
            return

        try:
            # Get available commands for this session
            commands = self.command_bridge.to_available_commands(self.agent.context)

            # Create update notification
            update = AvailableCommandsUpdate(
                sessionUpdate="available_commands_update",
                availableCommands=commands,
            )

            # Send to client
            notification = SessionNotification(sessionId=self.session_id, update=update)
            await self.client.sessionUpdate(notification)
            msg = "Sent %s available commands for session %s"
            logger.debug(msg, len(commands), self.session_id)

        except Exception:
            msg = "Failed to send available commands update for session %s"
            logger.exception(msg, self.session_id)

    async def update_available_commands(self, commands: list[AvailableCommand]) -> None:
        """Update and broadcast new command list.

        Args:
            commands: New list of available commands
        """
        try:
            # Create update notification
            update = AvailableCommandsUpdate(
                sessionUpdate="available_commands_update",
                availableCommands=commands,
            )

            # Send to client
            notification = SessionNotification(sessionId=self.session_id, update=update)
            await self.client.sessionUpdate(notification)

            logger.debug("Updated available commands for session %s", self.session_id)

        except Exception:
            msg = "Failed to update available commands for session %s"
            logger.exception(msg, self.session_id)


class ACPSessionManager:
    """Manages multiple ACP sessions and their lifecycle.

    Provides centralized management of ACP sessions, including:
    - Session creation and initialization
    - Session lookup and retrieval
    - Session cleanup and resource management
    - Agent instance management
    """

    def __init__(self, command_bridge: ACPCommandBridge) -> None:
        """Initialize session manager.

        Args:
            command_bridge: Optional command bridge for slash commands
        """
        self._sessions: dict[str, ACPSession] = {}
        self._lock = asyncio.Lock()
        self.command_bridge = command_bridge
        self._command_update_task: asyncio.Task | None = None

        # Register for command update notifications
        if command_bridge:
            command_bridge.register_update_callback(self._on_commands_updated)

        logger.info("Initialized ACP session manager")

    async def create_session(
        self,
        agent: Agent[Any],
        cwd: str,
        client: Client,
        mcp_servers: list[McpServer] | None = None,
        session_id: str | None = None,
        max_turn_requests: int = 50,
        max_tokens: int | None = None,
    ) -> str:
        """Create a new ACP session.

        Args:
            agent: llmling agent instance for the session
            cwd: Working directory for the session
            client: External library Client interface
            mcp_servers: Optional MCP server configurations
            session_id: Optional specific session ID (generated if None)
            max_turn_requests: Maximum model requests per turn
            max_tokens: Maximum tokens per turn (if None, no limit)

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
                max_turn_requests=max_turn_requests,
                max_tokens=max_tokens,
                command_bridge=self.command_bridge,
            )

            # Initialize MCP servers if any are provided
            await session.initialize_mcp_servers()

            # Store session
            self._sessions[session_id] = session

            # Announce available slash commands to client
            await session.send_available_commands_update()

            logger.info("Created ACP session %s for agent %s", session_id, agent.name)
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

    def _on_commands_updated(self) -> None:
        """Handle command updates by notifying all active sessions."""
        # Schedule async task to update all sessions
        task = asyncio.create_task(self._update_all_sessions_commands())
        # Store reference to prevent garbage collection
        self._command_update_task = task

    async def _update_all_sessions_commands(self) -> None:
        """Update available commands for all active sessions."""
        async with self._lock:
            for session in self._sessions.values():
                try:
                    await session.send_available_commands_update()
                except Exception:
                    logger.exception(
                        "Failed to update commands for session %s", session.session_id
                    )
