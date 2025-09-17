"""ACP (Agent Client Protocol) server implementation using external acp library.

This module provides the core ACP server that exposes llmling agents through the
standard Agent Client Protocol using the external acp library for robust
JSON-RPC 2.0 communication over stdio streams.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, overload

from acp import Agent as ACPAgent, AgentSideConnection
from acp.schema import (
    AgentCapabilities,
    AuthenticateRequest,
    CancelNotification,
    InitializeRequest,
    InitializeResponse,
    LoadSessionRequest,
    NewSessionRequest,
    NewSessionResponse,
    PromptCapabilities,
    PromptRequest,
    PromptResponse,
)
from acp.stdio import stdio_streams

from llmling_agent.log import get_logger
from llmling_agent_acp.session import ACPSessionManager
from llmling_agent_acp.wrappers import DefaultACPClient


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling_agent import Agent
    from llmling_agent_acp.wrappers import ACPClientInterface

logger = get_logger(__name__)


class LLMlingACPAgent(ACPAgent):
    """Implementation of ACP Agent protocol interface for llmling agents.

    This class implements the external library's Agent protocol interface,
    bridging llmling agents with the standard ACP JSON-RPC protocol.

    Protocol Compliance:
    - Explicitly inherits from acp.Agent protocol for type safety
    - Uses camelCase method names as required by ACP specification
    - Implements all required methods: initialize, newSession, loadSession,
      authenticate, prompt, cancel
    """

    def __init__(
        self,
        connection: AgentSideConnection,
        agents: dict[str, Agent],
        *,
        session_support: bool = True,
        file_access: bool = False,
        terminal_access: bool = False,
        client: ACPClientInterface | None = None,
    ) -> None:
        """Initialize ACP agent implementation.

        Args:
            connection: ACP connection for client communication
            agents: Dictionary of available llmling agents
            session_support: Whether agent supports session loading
            file_access: Whether agent can access filesystem
            terminal_access: Whether agent can use terminal
            client: Optional client interface for operations
        """
        self.connection = connection
        self.agents = agents
        self.session_support = session_support
        self.file_access = file_access
        self.terminal_access = terminal_access
        self.client = client or DefaultACPClient(allow_file_operations=file_access)

        # Session management
        self.session_manager = ACPSessionManager()

        # Track initialization state
        self._initialized = False
        self._protocol_version = 1

        logger.info("Created ACP agent implementation with %d agents", len(agents))

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        """Initialize the agent and negotiate capabilities."""
        try:
            logger.info("Initializing ACP agent implementation")

            # Store protocol version
            self._protocol_version = min(params.protocolVersion, 1)

            # Build agent capabilities
            prompt_caps = PromptCapabilities(
                audio=False,  # llmling-agent doesn't support audio by default
                embeddedContext=True,  # Support embedded resources
                image=False,  # Image support depends on model
            )

            agent_caps = AgentCapabilities(
                loadSession=self.session_support,
                promptCapabilities=prompt_caps,
            )

            # Mark as initialized
            self._initialized = True

            response = InitializeResponse(
                protocolVersion=self._protocol_version,
                agentCapabilities=agent_caps,
                authMethods=[],  # No authentication methods by default
            )

            logger.info("ACP agent implementation initialized successfully")
        except Exception:
            logger.exception("Failed to initialize ACP agent implementation")
            raise
        else:
            return response

    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse:
        """Create a new session."""
        if not self._initialized:
            msg = "Agent not initialized"
            raise RuntimeError(msg)

        try:
            logger.info("Creating new session")

            # Use the first available agent for now
            # In the future, we might want to support agent selection
            if not self.agents:
                msg = "No agents available"
                raise RuntimeError(msg)  # noqa: TRY301

            agent = next(iter(self.agents.values()))

            # Create session through session manager
            session_id = await self.session_manager.create_session(
                agent=agent,
                cwd=params.cwd,
                client=self.client,
                mcp_servers=params.mcpServers,
            )

            response = NewSessionResponse(sessionId=session_id)
            logger.info("Created session %s", session_id)

        except Exception:
            logger.exception("Failed to create new session")
            raise
        else:
            return response

    async def loadSession(self, params: LoadSessionRequest) -> None:
        """Load an existing session."""
        if not self._initialized:
            msg = "Agent not initialized"
            raise RuntimeError(msg)

        if not self.session_support:
            msg = "Session loading not supported"
            raise RuntimeError(msg)

        try:
            logger.info("Loading session %s", params.sessionId)

            # Get existing session
            session = await self.session_manager.get_session(params.sessionId)
            if not session:
                msg = f"Session {params.sessionId} not found"
                raise ValueError(msg)  # noqa: TRY301

            # Update session configuration if needed
            # This could involve updating the working directory, MCP servers, etc.
            # For now, we'll just log the load operation

            logger.info("Loaded session %s successfully", params.sessionId)

        except Exception:
            logger.exception("Failed to load session %s", params.sessionId)
            raise

    async def authenticate(self, params: AuthenticateRequest) -> None:
        """Authenticate with the agent."""
        # Basic implementation - no authentication required by default
        logger.info("Authentication requested with method %s", params.methodId)

        # In a real implementation, you might validate credentials here
        # For now, we'll just accept all authentication attempts

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        """Process a prompt request."""
        if not self._initialized:
            msg = "Agent not initialized"
            raise RuntimeError(msg)

        try:
            logger.info("Processing prompt for session %s", params.sessionId)

            # Get session
            session = await self.session_manager.get_session(params.sessionId)
            if not session:
                msg = f"Session {params.sessionId} not found"
                raise ValueError(msg)  # noqa: TRY301

            # Process prompt and stream responses
            response_sent = False
            async for notification in session.process_prompt(params.prompt):
                from acp.schema import SessionNotification as ACPSessionNotification

                acp_notification = ACPSessionNotification.model_validate(
                    notification.model_dump()
                )
                await self.connection.sessionUpdate(acp_notification)
                response_sent = True

            # Return completion status
            return PromptResponse(
                stopReason="completed" if response_sent else "no_output"
            )

        except Exception as e:
            logger.exception("Failed to process prompt for session %s", params.sessionId)

            # Send error as session update if possible
            from llmling_agent_acp.converters import to_session_updates

            error_updates = to_session_updates(
                f"Error processing prompt: {e}", params.sessionId
            )
            for update in error_updates:
                from acp.schema import SessionNotification as ACPSessionNotification

                acp_update = ACPSessionNotification.model_validate(update.model_dump())
                await self.connection.sessionUpdate(acp_update)

            return PromptResponse(stopReason="error")

    async def cancel(self, params: CancelNotification) -> None:
        """Cancel operations for a session."""
        try:
            logger.info("Cancelling operations for session %s", params.sessionId)

            # Get session
            session = await self.session_manager.get_session(params.sessionId)
            if session:
                # In a real implementation, you might stop ongoing operations
                # For now, we'll just log the cancellation
                logger.info("Cancelled operations for session %s", params.sessionId)
            else:
                logger.warning("Session %s not found for cancellation", params.sessionId)

        except Exception:
            logger.exception("Failed to cancel session %s", params.sessionId)


class ACPServer:
    """ACP (Agent Client Protocol) server for llmling-agent using external library.

    Provides a bridge between llmling-agent's Agent system and the standard ACP
    JSON-RPC protocol using the external acp library for robust communication.
    """

    def __init__(
        self,
        *,
        client: ACPClientInterface | None = None,
    ) -> None:
        """Initialize ACP server.

        Args:
            client: ACP client interface for operations (DefaultACPClient if None)
        """
        self._client = client or DefaultACPClient(allow_file_operations=True)

        # Agent management
        self._agents: dict[str, Agent] = {}
        self._running = False

        # Server configuration
        self._session_support = True
        self._file_access = False
        self._terminal_access = False

    @overload
    def agent(
        self,
        agent: Agent,
        *,
        name: str | None = None,
        session_support: bool = True,
        file_access: bool = False,
        terminal_access: bool = False,
    ) -> Agent: ...

    @overload
    def agent(
        self,
        agent: None = None,
        *,
        name: str | None = None,
        session_support: bool = True,
        file_access: bool = False,
        terminal_access: bool = False,
    ) -> Callable[[Agent | Callable[[], Agent]], Agent]: ...

    def agent(
        self,
        agent: Agent | None = None,
        *,
        name: str | None = None,
        session_support: bool = True,
        file_access: bool = False,
        terminal_access: bool = False,
    ) -> Agent | Callable[[Agent | Callable[[], Agent]], Agent]:
        """Register a llmling Agent as an ACP agent.

        Args:
            agent: llmling Agent to register (can be provided via decorator)
            name: Optional name override for the agent
            session_support: Whether agent supports session loading
            file_access: Whether agent can access filesystem
            terminal_access: Whether agent can use terminal

        Returns:
            Decorator function or registered agent function

        Example:
            ```python
            server = ACPServer()

            # Method 1: Direct registration
            my_agent = Agent("my-agent", model="gpt-4")
            server.agent(my_agent, name="chat_agent")

            # Method 2: Decorator
            @server.agent(name="expert_agent", file_access=True)
            def create_expert():
                return Agent("expert", model="gpt-4", tools=["file_tools"])
            ```
        """

        def decorator(agent_or_factory: Agent | Callable[[], Agent]) -> Agent:
            # Handle different input types
            if callable(agent_or_factory) and not hasattr(agent_or_factory, "run"):
                # It's a factory function - call it to get the agent
                actual_agent: Agent = agent_or_factory()
            else:
                # It's an agent instance - cast to correct type
                actual_agent = agent_or_factory  # type: ignore[assignment]

            agent_name = name or actual_agent.name

            # Store references - cast to bypass generic type checking
            self._agents[agent_name] = actual_agent  # type: ignore[assignment]

            # Update server capabilities based on agent requirements
            if file_access:
                self._file_access = True
            if terminal_access:
                self._terminal_access = True

            logger.info("Registered ACP agent %s", agent_name)
            return actual_agent

        # Handle direct registration vs decorator usage
        if agent is not None:
            return decorator(agent)  # type: ignore[return-value]
        return decorator

    @classmethod
    async def from_config(
        cls,
        config_path: str | Path,
        **kwargs: Any,
    ) -> ACPServer:
        """Create ACP server from existing llmling-agent configuration.

        Args:
            config_path: Path to llmling-agent YAML config file
            **kwargs: Additional server initialization parameters

        Returns:
            Configured ACP server instance with agents from config
        """
        from llmling_agent.models.manifest import AgentsManifest

        config_str = Path(config_path).read_text()
        manifest = AgentsManifest.from_yaml(config_str)

        # Create server
        server = cls(**kwargs)

        # Create agent pool from manifest to get configured agents
        async with manifest.pool as pool:
            # Register each agent from the config as ACP agent
            for agent_name in manifest.agents:
                agent = pool.get_agent(agent_name)

                # Determine capabilities based on agent tools
                file_access = any("file" in str(tool).lower() for tool in agent.tools)

                # Register with ACP server
                server.agent(
                    agent,
                    name=agent_name,
                    session_support=True,
                    file_access=file_access,
                    terminal_access=False,  # Conservative default
                )

        return server

    def get_agent(self, name: str) -> Agent | None:
        """Get registered agent by name."""
        return self._agents.get(name)

    def list_agents(self) -> list[str]:
        """List all registered agent names."""
        return list(self._agents.keys())

    async def run(self) -> None:
        """Run the ACP server using external library."""
        if self._running:
            return

        try:
            self._running = True

            if not self._agents:
                msg = "No agents registered"
                raise RuntimeError(msg)  # noqa: TRY301

            logger.info("Starting ACP server with %d agents on stdio", len(self._agents))

            # Create stdio streams
            reader, writer = await stdio_streams()

            # Create agent factory function for external library
            def create_acp_agent(connection: AgentSideConnection) -> ACPAgent:
                return LLMlingACPAgent(
                    connection=connection,
                    agents=self._agents,
                    session_support=self._session_support,
                    file_access=self._file_access,
                    terminal_access=self._terminal_access,
                    client=self._client,
                )

            # Create ACP connection using external library
            # AgentSideConnection expects (factory, input_stream, output_stream)
            # where input_stream writes to peer and output_stream reads from peer
            AgentSideConnection(create_acp_agent, writer, reader)

            logger.info(
                "ACP server started with protocol features: "
                "file_access=%s, terminal_access=%s, session_support=%s",
                self._file_access,
                self._terminal_access,
                self._session_support,
            )

            # Keep the connection alive
            try:
                while self._running:
                    await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("ACP server shutdown requested")

        except Exception:
            logger.exception("Error running ACP server")
            raise
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown the ACP server and cleanup resources."""
        if not self._running:
            return

        self._running = False

        logger.info("Shutting down ACP server")

        # Cleanup agents
        for agent in self._agents.values():
            try:
                if hasattr(agent, "__aexit__"):
                    await agent.__aexit__(None, None, None)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to cleanup agent %s: %s", agent.name, e)

        # Clear registrations
        self._agents.clear()

        logger.info("ACP server shutdown complete")

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()
