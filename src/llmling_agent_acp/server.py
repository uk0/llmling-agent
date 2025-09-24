"""ACP (Agent Client Protocol) server implementation for llmling-agent.

This module provides the main server class for exposing llmling agents via
the Agent Client Protocol, using the external acp library for robust
JSON-RPC 2.0 communication over stdio streams.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from slashed import CommandStore

from acp import Agent as ACPAgent, AgentSideConnection, create_session_model_state
from acp.schema import (
    AgentCapabilities,
    InitializeResponse,
    LoadSessionResponse,
    McpCapabilities,
    NewSessionResponse,
    PromptCapabilities,
    PromptResponse,
    SessionMode,
    SessionModeState,
    SetSessionModelRequest,
    SetSessionModelResponse,
    SetSessionModeRequest,
    SetSessionModeResponse,
)
from acp.stdio import stdio_streams
from llmling_agent.log import get_logger
from llmling_agent.models.manifest import AgentsManifest
from llmling_agent_acp.command_bridge import ACPCommandBridge
from llmling_agent_acp.converters import to_session_updates
from llmling_agent_acp.session import ACPSessionManager
from llmling_agent_acp.wrappers import DefaultACPClient
from llmling_agent_commands import get_commands


if TYPE_CHECKING:
    from acp.schema import (
        AuthenticateRequest,
        CancelNotification,
        InitializeRequest,
        LoadSessionRequest,
        NewSessionRequest,
        PromptRequest,
        SetSessionModelRequest,
        SetSessionModeRequest,
    )
    from llmling_agent import Agent, AgentPool
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

    PROTOCOL_VERSION = 1

    def __init__(
        self,
        connection: AgentSideConnection,
        agent_pool: AgentPool,
        *,
        session_support: bool = True,
        file_access: bool = False,
        terminal_access: bool = False,
        client: ACPClientInterface | None = None,
        max_turn_requests: int = 50,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize ACP agent implementation.

        Args:
            connection: ACP connection for client communication
            agent_pool: AgentPool containing available agents
            session_support: Whether agent supports session loading
            file_access: Whether agent can access filesystem
            terminal_access: Whether agent can use terminal
            client: Optional client interface for operations
            max_turn_requests: Maximum model requests per turn
            max_tokens: Maximum tokens per turn (if None, no limit)
        """
        self.connection = connection
        self.agent_pool = agent_pool
        self.session_support = session_support
        self.file_access = file_access
        self.terminal_access = terminal_access
        self.client = client or DefaultACPClient(allow_file_operations=file_access)
        self.max_turn_requests = max_turn_requests
        self.max_tokens = max_tokens
        command_store = CommandStore(enable_system_commands=True)
        command_store._initialize_sync()  # Ensure store is initialized

        for command in get_commands():
            command_store.register_command(command)

        self.command_bridge = ACPCommandBridge(command_store)
        self.session_manager = ACPSessionManager(command_bridge=self.command_bridge)

        self._initialized = False
        agent_count = len(self.agent_pool.agents)
        logger.info("Created ACP agent implementation with %d agents", agent_count)

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        """Initialize the agent and negotiate capabilities."""
        logger.info("Initializing ACP agent implementation")
        version = min(params.protocol_version, self.PROTOCOL_VERSION)
        prompt_caps = PromptCapabilities(audio=True, embedded_context=True, image=True)
        mcp_caps = McpCapabilities(http=False, sse=False)
        caps = AgentCapabilities(
            load_session=self.session_support,
            prompt_capabilities=prompt_caps,
            mcp_capabilities=mcp_caps,
        )

        self._initialized = True
        response = InitializeResponse(protocol_version=version, agent_capabilities=caps)

        logger.info("ACP agent implementation initialized successfully: %s", response)
        return response

    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse:
        """Create a new session."""
        if not self._initialized:
            msg = "Agent not initialized"
            raise RuntimeError(msg)

        try:
            logger.info("Creating new session")
            agent_names = list(self.agent_pool.agents.keys())
            logger.info("Available agents: %s", agent_names)
            if not agent_names:
                logger.error("No agents available for session creation")
                msg = "No agents available"
                raise RuntimeError(msg)  # noqa: TRY301

            default_agent_name = agent_names[0]  # Use the first agent as default
            logger.info("Using agent %s as default for new session", default_agent_name)

            # Create session through session manager (pass the pool, not individual agent)
            session_id = await self.session_manager.create_session(
                agent_pool=self.agent_pool,
                default_agent_name=default_agent_name,
                cwd=params.cwd,
                client=self.client,
                mcp_servers=params.mcp_servers,
                max_turn_requests=self.max_turn_requests,
                max_tokens=self.max_tokens,
            )

            # Create session modes from available agents

            available_modes = [
                SessionMode(
                    id=agent_name,
                    name=self.agent_pool.get_agent(agent_name).name,
                    description=(
                        self.agent_pool.get_agent(agent_name).description
                        or f"Switch to {self.agent_pool.get_agent(agent_name).name} agent"
                    ),
                )
                for agent_name in agent_names
            ]

            modes = SessionModeState(
                current_mode_id=default_agent_name, available_modes=available_modes
            )

            # Get model information from the default agent
            session = await self.session_manager.get_session(session_id)
            if session and session.agent:
                # For now, create model state with single current model
                # TODO: Get list of available models from agent provider
                current_model = session.agent.model_name
                if current_model:
                    models = create_session_model_state([current_model], current_model)
                else:
                    models = None
            else:
                models = None

            response = NewSessionResponse(
                session_id=session_id, modes=modes, models=models
            )
            msg = "Created session %s with %d available agents"
            logger.info(msg, session_id, len(available_modes))
            session = await self.session_manager.get_session(session_id)
            assert session
            logger.debug("About to send available commands update")
            await session.send_available_commands_update()

        except Exception:
            logger.exception("Failed to create new session")
            raise
        else:
            return response

    async def loadSession(self, params: LoadSessionRequest) -> LoadSessionResponse:
        """Load an existing session."""
        if not self._initialized:
            msg = "Agent not initialized"
            raise RuntimeError(msg)

        try:
            # Get the session
            session = await self.session_manager.get_session(params.session_id)
            if not session:
                logger.warning("Session %s not found", params.session_id)
                return LoadSessionResponse()

            # Get model information
            current_model = session.agent.model_name if session.agent else None
            if current_model:
                models = create_session_model_state([current_model], current_model)
            else:
                models = None

            return LoadSessionResponse(models=models)
        except Exception:
            logger.exception("Failed to load session %s", params.session_id)
            return LoadSessionResponse()

    async def authenticate(self, params: AuthenticateRequest) -> None:
        """Authenticate with the agent."""
        logger.info("Authentication requested with method %s", params.method_id)

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        """Process a prompt request."""
        if not self._initialized:
            msg = "Agent not initialized"
            raise RuntimeError(msg)

        try:
            logger.info("Processing prompt for session %s", params.session_id)
            session = await self.session_manager.get_session(params.session_id)
            if not session:
                msg = f"Session {params.session_id} not found"
                raise ValueError(msg)  # noqa: TRY301

            # Process prompt and stream responses
            stop_reason = "end_turn"  # Default stop reason
            async for result in session.process_prompt(params.prompt):
                if isinstance(result, str):
                    stop_reason = result
                    break
                msg = "Sending sessionUpdate notification: %s"
                logger.info(
                    msg,
                    result.model_dump_json(
                        exclude_none=True, by_alias=True, exclude_defaults=False
                    ),
                )
                await self.connection.sessionUpdate(result)

            # Return the actual stop reason from the session
            response = PromptResponse(stop_reason=stop_reason)
            msg = "Returning PromptResponse: %s"
            logger.info(msg, response.model_dump_json(exclude_none=True, by_alias=True))
        except Exception as e:
            logger.exception("Failed to process prompt for session %s", params.session_id)
            msg = f"Error processing prompt: {e}"
            error_updates = to_session_updates(msg, params.session_id)
            for update in error_updates:
                try:
                    await self.connection.sessionUpdate(update)
                except Exception:
                    logger.exception("Failed to send error update")

            return PromptResponse(stop_reason="refusal")
        else:
            return response

    async def cancel(self, params: CancelNotification) -> None:
        """Cancel operations for a session."""
        try:
            logger.info("Cancelling session %s", params.session_id)

            # Get session and cancel it
            session = await self.session_manager.get_session(params.session_id)
            if session:
                session.cancel()
                logger.info("Cancelled operations for session %s", params.session_id)
            else:
                logger.warning("Session %s not found for cancellation", params.session_id)

        except Exception:
            logger.exception("Failed to cancel session %s", params.session_id)

    async def extMethod(self, method: str, params: dict) -> dict:
        return {"example": "response"}

    async def extNotification(self, method: str, params: dict) -> None:
        return None

    async def setSessionMode(
        self, params: SetSessionModeRequest
    ) -> SetSessionModeResponse | None:
        """Set the session mode (switch active agent).

        The mode ID corresponds to the agent name in the pool.
        """
        try:
            # Get session and switch active agent
            session = await self.session_manager.get_session(params.session_id)
            if not session:
                logger.warning("Session %s not found for mode switch", params.session_id)
                return None

            # Validate agent exists in pool
            if not self.agent_pool or params.mode_id not in self.agent_pool.agents:
                logger.error("Agent %s not found in pool", params.mode_id)
                return None

            # Switch the active agent in the session
            await session.switch_active_agent(params.mode_id)

            logger.info(
                "Switched session %s to agent %s", params.session_id, params.mode_id
            )
            return SetSessionModeResponse()

        except Exception:
            logger.exception("Failed to set session mode for %s", params.session_id)
            return None

    async def setSessionModel(
        self, params: SetSessionModelRequest
    ) -> SetSessionModelResponse | None:
        """Set the session model.

        Changes the model for the active agent in the session.
        """
        try:
            # Get session and active agent
            session = await self.session_manager.get_session(params.session_id)
            if not session:
                logger.warning("Session %s not found for model switch", params.session_id)
                return None

            # Get the active agent from the session
            active_agent = session.agent
            if not active_agent:
                logger.warning("No active agent in session %s", params.session_id)
                return None

            # Set the model on the active agent
            active_agent.set_model(params.model_id)

            logger.info("Set model %s for session %s", params.model_id, params.session_id)
            return SetSessionModelResponse()

        except Exception:
            logger.exception("Failed to set session model for %s", params.session_id)
            return None


class ACPServer:
    """ACP (Agent Client Protocol) server for llmling-agent using external library.

    Provides a bridge between llmling-agent's Agent system and the standard ACP
    JSON-RPC protocol using the external acp library for robust communication.
    """

    def __init__(
        self,
        *,
        client: ACPClientInterface | None = None,
        max_turn_requests: int = 50,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize ACP server.

        Args:
            client: ACP client interface for operations (DefaultACPClient if None)
            max_turn_requests: Maximum model requests per turn
            max_tokens: Maximum tokens per turn (if None, no limit)
        """
        self._client = client or DefaultACPClient(allow_file_operations=True)

        # Agent pool management
        self._agent_pool: AgentPool | None = None
        self._running = False

        # Server configuration
        self._session_support = True
        self._file_access = False
        self._terminal_access = False
        self._max_turn_requests = max_turn_requests
        self._max_tokens = max_tokens

    def set_agent_pool(
        self,
        agent_pool: AgentPool,
        *,
        session_support: bool = True,
        file_access: bool = False,
        terminal_access: bool = False,
    ) -> None:
        """Set the agent pool for this ACP server.

        Args:
            agent_pool: AgentPool containing available agents
            session_support: Enable session loading support
            file_access: Enable file system access
            terminal_access: Enable terminal access
        """
        self._agent_pool = agent_pool
        self._session_support = session_support
        self._file_access = file_access
        self._terminal_access = terminal_access

        logger.info("Set agent pool with %d agents", len(agent_pool.agents))

    @property
    def agent_pool(self) -> AgentPool | None:
        """Get the current agent pool."""
        return self._agent_pool

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
            Configured ACP server instance with agent pool from config
        """
        config_str = Path(config_path).read_text()
        manifest = AgentsManifest.from_yaml(config_str)

        # Create server
        server = cls(**kwargs)

        # Store the agent pool - server context manager will handle lifecycle
        server._agent_pool = manifest.pool

        # Set up the agent pool with capabilities
        server.set_agent_pool(
            agent_pool=server._agent_pool,
            session_support=True,
            file_access=True,
            terminal_access=False,  # Conservative default
        )

        agent_names = list(server._agent_pool.agents.keys())
        logger.info("Created ACP server with agent pool containing: %s", agent_names)

        return server

    def get_agent(self, name: str) -> Agent | None:
        """Get agent by name from the pool."""
        if not self._agent_pool:
            return None
        return self._agent_pool.get_agent(name)

    def list_agents(self) -> list[str]:
        """List all available agent names."""
        if not self._agent_pool:
            return []
        return list(self._agent_pool.agents.keys())

    async def run(self) -> None:
        """Run the ACP server using external library."""
        if self._running:
            return

        try:
            self._running = True

            if not self._agent_pool:
                logger.error("No agent pool available - cannot start server")
                msg = "No agent pool available"
                raise RuntimeError(msg)  # noqa: TRY301

            agent_names = list(self._agent_pool.agents.keys())
            msg = "Starting ACP server with %d agents on stdio: %s"
            logger.info(msg, len(agent_names), agent_names)
            # Create stdio streams
            reader, writer = await stdio_streams()

            # Create agent factory function for external library
            def create_acp_agent(connection: AgentSideConnection) -> ACPAgent:
                if not self._agent_pool:
                    msg = "Agent pool not initialized"
                    raise RuntimeError(msg)  # noqa: TRY301
                return LLMlingACPAgent(
                    connection=connection,
                    agent_pool=self._agent_pool,
                    session_support=self._session_support,
                    file_access=self._file_access,
                    terminal_access=self._terminal_access,
                    client=self._client,
                    max_turn_requests=self._max_turn_requests,
                    max_tokens=self._max_tokens,
                )

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

        # Cleanup agent pool
        if self._agent_pool:
            try:
                await self._agent_pool.__aexit__(None, None, None)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to cleanup agent pool: %s", e)

        self._agent_pool = None
        logger.info("ACP server shutdown complete")

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        # Initialize agent pool if present
        if self._agent_pool:
            await self._agent_pool.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()
