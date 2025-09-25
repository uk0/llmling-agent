"""ACP (Agent Client Protocol) Agent implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slashed import CommandStore

from acp import Agent as ACPAgent, create_session_model_state
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
from llmling_agent.log import get_logger
from llmling_agent.utils.tasks import TaskManagerMixin
from llmling_agent_acp.command_bridge import ACPCommandBridge
from llmling_agent_acp.converters import to_session_updates
from llmling_agent_acp.session import ACPSessionManager
from llmling_agent_acp.tools import get_filesystem_tools, get_terminal_tools
from llmling_agent_commands import get_commands


if TYPE_CHECKING:
    from tokonomics.model_discovery.model_info import ModelInfo

    from acp import AgentSideConnection, Client
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
    from llmling_agent import AgentPool
    from llmling_agent_providers.base import UsageLimits

logger = get_logger(__name__)


class LLMlingACPAgent(ACPAgent):
    """Implementation of ACP Agent protocol interface for llmling agents.

    This class implements the external library's Agent protocol interface,
    bridging llmling agents with the standard ACP JSON-RPC protocol.
    """

    PROTOCOL_VERSION = 1

    def __init__(
        self,
        connection: AgentSideConnection,
        agent_pool: AgentPool[Any],
        *,
        available_models: list[ModelInfo] | None = None,
        session_support: bool = True,
        file_access: bool = False,
        terminal_access: bool = False,
        client: Client | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> None:
        """Initialize ACP agent implementation.

        Args:
            connection: ACP connection for client communication
            agent_pool: AgentPool containing available agents
            available_models: List of available tokonomics ModelInfo objects
            session_support: Whether agent supports session loading
            file_access: Whether agent can access filesystem
            terminal_access: Whether agent can use terminal
            client: Optional client interface for operations
            usage_limits: Optional usage limits for model requests and tokens
        """
        self.connection = connection
        self.agent_pool = agent_pool
        self.available_models = available_models or []
        self.session_support = session_support
        self.file_access = file_access
        self.terminal_access = terminal_access
        self.client: Client = client or connection
        self.usage_limits = usage_limits

        # Terminal support - client-side execution via ACP protocol
        # Filesystem support - client-side file operations via ACP protocol

        command_store = CommandStore(enable_system_commands=True)
        command_store._initialize_sync()  # Ensure store is initialized

        for command in get_commands():
            command_store.register_command(command)

        self.command_bridge = ACPCommandBridge(command_store)
        self.session_manager = ACPSessionManager(command_bridge=self.command_bridge)
        self.tasks = TaskManagerMixin()

        self._initialized = False
        agent_count = len(self.agent_pool.agents)
        logger.info("Created ACP agent implementation with %d agents", agent_count)

        # Register terminal tools with agents if terminal access is enabled
        if self.terminal_access:
            self._register_terminal_tools_with_agents()

        # Register filesystem tools with agents
        self._register_filesystem_tools_with_agents()

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        """Initialize the agent and negotiate capabilities."""
        logger.info("Initializing ACP agent implementation")
        version = min(params.protocol_version, self.PROTOCOL_VERSION)
        prompt_caps = PromptCapabilities(audio=True, embedded_context=True, image=True)
        mcp_caps = McpCapabilities(http=True, sse=True)
        caps = AgentCapabilities(
            load_session=self.session_support,
            prompt_capabilities=prompt_caps,
            mcp_capabilities=mcp_caps,
        )

        self._initialized = True
        response = InitializeResponse(protocol_version=version, agent_capabilities=caps)
        logger.info("ACP agent implementation initialized successfully: %s", response)
        return response

    async def new_session(self, params: NewSessionRequest) -> NewSessionResponse:
        """Create a new session."""
        if not self._initialized:
            msg = "Agent not initialized"
            raise RuntimeError(msg)

        try:
            agent_names = list(self.agent_pool.agents.keys())
            if not agent_names:
                logger.error("No agents available for session creation")
                msg = "No agents available"
                raise RuntimeError(msg)  # noqa: TRY301

            default_name = agent_names[0]  # Use the first agent as default
            msg = "Creating new session. Available agents: %s. Default agent: %s"
            logger.info(msg, agent_names, default_name)
            # Create session through session manager (pass the pool, not individual agent)
            session_id = await self.session_manager.create_session(
                agent_pool=self.agent_pool,
                default_agent_name=default_name,
                cwd=params.cwd,
                client=self.client,
                mcp_servers=params.mcp_servers,
                usage_limits=self.usage_limits,
            )

            # Create session modes from available agents
            modes = [
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

            state = SessionModeState(current_mode_id=default_name, available_modes=modes)
            # Get model information from the default agent
            session = await self.session_manager.get_session(session_id)
            if session:
                current_model = session.agent.model_name
                model_ids = [m.pydantic_ai_id for m in self.available_models]
                models = create_session_model_state(model_ids, current_model)
            else:
                models = None

            response = NewSessionResponse(
                session_id=session_id, modes=state, models=models
            )
            msg = "Created session %s with %d available agents"
            logger.info(msg, session_id, len(modes))

        except Exception:
            logger.exception("Failed to create new session")
            raise
        else:
            # Schedule available commands update after session response is returned
            session = await self.session_manager.get_session(session_id)
            if session:
                # Schedule task to run after response is sent
                coro = session.send_available_commands_update()
                self.tasks.create_task(coro, name=f"send_commands_update_{session_id}")
            return response

    async def load_session(self, params: LoadSessionRequest) -> LoadSessionResponse:
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
            model_ids = [m.pydantic_ai_id for m in self.available_models]
            models = create_session_model_state(model_ids, current_model)

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
                logger.info(msg, result.model_dump_json(exclude_none=True, by_alias=True))
                await self.connection.session_update(result)

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
                    await self.connection.session_update(update)
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

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"example": "response"}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None

    async def set_session_mode(
        self, params: SetSessionModeRequest
    ) -> SetSessionModeResponse | None:
        """Set the session mode (switch active agent).

        The mode ID corresponds to the agent name in the pool.
        """
        try:
            session = await self.session_manager.get_session(params.session_id)
            if not session:
                logger.warning("Session %s not found for mode switch", params.session_id)
                return None

            # Validate agent exists in pool
            if not self.agent_pool or params.mode_id not in self.agent_pool.agents:
                logger.error("Agent %s not found in pool", params.mode_id)
                return None

            await session.switch_active_agent(params.mode_id)
            msg = "Switched session %s to agent %s"
            logger.info(msg, params.session_id, params.mode_id)
            return SetSessionModeResponse()

        except Exception:
            logger.exception("Failed to set session mode for %s", params.session_id)
            return None

    async def set_session_model(
        self, params: SetSessionModelRequest
    ) -> SetSessionModelResponse | None:
        """Set the session model.

        Changes the model for the active agent in the session.
        """
        try:
            session = await self.session_manager.get_session(params.session_id)
            if not session:
                logger.warning("Session %s not found for model switch", params.session_id)
                return None
            session.agent.set_model(params.model_id)
            logger.info("Set model %s for session %s", params.model_id, params.session_id)
            return SetSessionModelResponse()
        except Exception:
            logger.exception("Failed to set session model for %s", params.session_id)
            return None

    def _register_terminal_tools_with_agents(self) -> None:
        """Register client-side terminal tools with all agents in the pool."""
        if not self.agent_pool or not self.agent_pool.agents:
            logger.debug("No agents in pool to register terminal tools with")
            return

        # Get terminal tools
        tools = get_terminal_tools(self)

        # Register tools with each agent in the pool
        registered_count = 0
        for agent_name, agent in self.agent_pool.agents.items():
            for tool in tools:
                agent.tools.register_tool(tool)
            registered_count += 1
            logger.debug(
                "Registered %d terminal tools with agent %s",
                len(tools),
                agent_name,
            )

        logger.info(
            "Registered terminal tools with %d agents (%d tools per agent)",
            registered_count,
            len(tools),
        )

    def _register_filesystem_tools_with_agents(self) -> None:
        """Register client-side filesystem tools with all agents in the pool."""
        if not self.agent_pool or not self.agent_pool.agents:
            logger.debug("No agents in pool to register filesystem tools with")
            return

        # Get filesystem tools
        tools = get_filesystem_tools(self)

        # Register tools with each agent in the pool
        registered_count = 0
        for agent_name, agent in self.agent_pool.agents.items():
            for tool in tools:
                agent.tools.register_tool(tool)
            registered_count += 1
            logger.debug(
                "Registered %d filesystem tools with agent %s",
                len(tools),
                agent_name,
            )

        logger.info(
            "Registered filesystem tools with %d agents (%d tools per agent)",
            registered_count,
            len(tools),
        )
