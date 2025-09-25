"""ACP (Agent Client Protocol) Agent implementation."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

from slashed import CommandStore

from acp import Agent as ACPAgent, create_session_model_state
from acp.schema import (
    AgentCapabilities,
    CreateTerminalRequest,
    EnvVariable,
    InitializeResponse,
    KillTerminalCommandRequest,
    LoadSessionResponse,
    McpCapabilities,
    NewSessionResponse,
    PromptCapabilities,
    PromptResponse,
    ReadTextFileRequest,
    ReleaseTerminalRequest,
    SessionMode,
    SessionModeState,
    SetSessionModelRequest,
    SetSessionModelResponse,
    SetSessionModeRequest,
    SetSessionModeResponse,
    TerminalOutputRequest,
    WaitForTerminalExitRequest,
    WriteTextFileRequest,
)
from llmling_agent.log import get_logger
from llmling_agent.tools.base import Tool
from llmling_agent.utils.tasks import TaskManagerMixin
from llmling_agent_acp.command_bridge import ACPCommandBridge
from llmling_agent_acp.converters import to_session_updates
from llmling_agent_acp.session import ACPSessionManager
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

        # Create client-side terminal tools
        tools = [
            Tool.from_callable(
                self._create_run_command_tool(),
                source="terminal",
                name_override="run_command",
            ),
            Tool.from_callable(
                self._create_get_command_output_tool(),
                source="terminal",
                name_override="get_command_output",
            ),
            Tool.from_callable(
                self._create_create_terminal_tool(),
                source="terminal",
                name_override="create_terminal",
            ),
            Tool.from_callable(
                self._create_wait_for_terminal_exit_tool(),
                source="terminal",
                name_override="wait_for_terminal_exit",
            ),
            Tool.from_callable(
                self._create_kill_terminal_tool(),
                source="terminal",
                name_override="kill_terminal",
            ),
            Tool.from_callable(
                self._create_release_terminal_tool(),
                source="terminal",
                name_override="release_terminal",
            ),
            Tool.from_callable(
                self._create_run_command_with_timeout_tool(),
                source="terminal",
                name_override="run_command_with_timeout",
            ),
        ]

        # Register tools with each agent in the pool
        for agent_name, agent in self.agent_pool.agents.items():
            for tool in tools:
                agent.tools.register_tool(tool)
            msg = "Registered %d terminal tools with agent %s"
            logger.debug(msg, len(tools), agent_name)

        msg = "Registered terminal tools with %d agents (%d tools per agent)"
        logger.info(msg, len(self.agent_pool.agents), len(tools))

    def _create_run_command_tool(self):
        """Create a tool that runs commands via the ACP client."""

        async def run_command(
            command: str,
            args: list[str] | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            session_id: str = "default_session",
        ) -> str:
            """Execute a command in the client's environment.

            Args:
                command: The command to execute
                args: Command arguments (optional)
                cwd: Working directory (optional)
                env: Environment variables (optional)
                session_id: Session ID for the request

            Returns:
                Combined stdout and stderr output, or error message
            """
            try:
                # Create terminal via client
                env_var = [EnvVariable(name=k, value=v) for k, v in (env or {}).items()]
                request = CreateTerminalRequest(
                    session_id=session_id,
                    command=command,
                    args=args or [],
                    cwd=cwd,
                    env=env_var,
                    output_byte_limit=1048576,
                )
                create_response = await self.connection.create_terminal(request)
                terminal_id = create_response.terminal_id
                wait_request = WaitForTerminalExitRequest(
                    session_id=session_id,
                    terminal_id=terminal_id,
                )
                await self.connection.wait_for_terminal_exit(wait_request)
                output_request = TerminalOutputRequest(
                    session_id=session_id,
                    terminal_id=terminal_id,
                )
                output_response = await self.connection.terminal_output(output_request)
                release_request = ReleaseTerminalRequest(
                    session_id=session_id,
                    terminal_id=terminal_id,
                )
                await self.connection.release_terminal(release_request)

                result = output_response.output
                if output_response.exit_status:
                    code = output_response.exit_status.exit_code
                    if code is not None:
                        result += f"\n[Command exited with code {code}]"
                    if output_response.exit_status.signal:
                        result += (
                            f"\n[Terminated by signal "
                            f"{output_response.exit_status.signal}]"
                        )

            except Exception as e:  # noqa: BLE001
                return f"Error executing command: {e}"
            else:
                return result

        return run_command

    def _create_get_command_output_tool(self):
        """Create a tool that gets output from a running command."""

        async def get_command_output(
            terminal_id: str,
            session_id: str = "default_session",
        ) -> str:
            """Get current output from a running command.

            Args:
                terminal_id: The terminal ID to get output from
                session_id: Session ID for the request

            Returns:
                Current command output
            """
            try:
                # Get output
                request = TerminalOutputRequest(
                    session_id=session_id,
                    terminal_id=terminal_id,
                )
                output_response = await self.connection.terminal_output(request)

                result = output_response.output
                if output_response.truncated:
                    result += "\n[Output was truncated]"
                if output_response.exit_status:
                    if (code := output_response.exit_status.exit_code) is not None:
                        result += f"\n[Exited with code {code}]"
                    if signal := output_response.exit_status.signal:
                        result += f"\n[Terminated by signal {signal}]"
                else:
                    result += "\n[Still running]"
            except Exception as e:  # noqa: BLE001
                return f"Error getting command output: {e}"
            else:
                return result

        return get_command_output

    def _create_create_terminal_tool(self):
        """Create a tool that creates a terminal and returns the terminal ID."""

        async def create_terminal(
            command: str,
            args: list[str] | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            output_byte_limit: int = 1048576,
            session_id: str = "default_session",
        ) -> str:
            """Create a terminal and start executing a command.

            Args:
                command: The command to execute
                args: Command arguments (optional)
                cwd: Working directory (optional)
                env: Environment variables (optional)
                output_byte_limit: Maximum output bytes to retain
                session_id: Session ID for the request

            Returns:
                Terminal ID for the created terminal
            """
            try:
                request = CreateTerminalRequest(
                    session_id=session_id,
                    command=command,
                    args=args or [],
                    cwd=cwd,
                    env=[EnvVariable(name=k, value=v) for k, v in (env or {}).items()],
                    output_byte_limit=output_byte_limit,
                )
                create_response = await self.connection.create_terminal(request)
            except Exception as e:  # noqa: BLE001
                return f"Error creating terminal: {e}"
            else:
                return create_response.terminal_id

        return create_terminal

    def _create_wait_for_terminal_exit_tool(self):
        """Create a tool that waits for a terminal to exit."""

        async def wait_for_terminal_exit(
            terminal_id: str,
            session_id: str = "default_session",
        ) -> str:
            """Wait for a terminal command to complete.

            Args:
                terminal_id: The terminal ID to wait for
                session_id: Session ID for the request

            Returns:
                Exit status information
            """
            try:
                request = WaitForTerminalExitRequest(
                    session_id=session_id,
                    terminal_id=terminal_id,
                )
                exit_response = await self.connection.wait_for_terminal_exit(request)

                result = f"Terminal {terminal_id} completed"
                if exit_response.exit_code is not None:
                    result += f" with exit code {exit_response.exit_code}"
                if exit_response.signal:
                    result += f" (terminated by signal {exit_response.signal})"
            except Exception as e:  # noqa: BLE001
                return f"Error waiting for terminal exit: {e}"
            else:
                return result

        return wait_for_terminal_exit

    def _create_kill_terminal_tool(self):
        """Create a tool that kills a running terminal command."""

        async def kill_terminal(
            terminal_id: str,
            session_id: str = "default_session",
        ) -> str:
            """Kill a running terminal command.

            Args:
                terminal_id: The terminal ID to kill
                session_id: Session ID for the request

            Returns:
                Success/failure message
            """
            try:
                request = KillTerminalCommandRequest(
                    session_id=session_id,
                    terminal_id=terminal_id,
                )
                await self.connection.kill_terminal(request)
            except Exception as e:  # noqa: BLE001
                return f"Error killing terminal: {e}"
            else:
                return f"Terminal {terminal_id} killed successfully"

        return kill_terminal

    def _create_release_terminal_tool(self):
        """Create a tool that releases terminal resources."""

        async def release_terminal(
            terminal_id: str,
            session_id: str = "default_session",
        ) -> str:
            """Release a terminal and free its resources.

            Args:
                terminal_id: The terminal ID to release
                session_id: Session ID for the request

            Returns:
                Success/failure message
            """
            try:
                request = ReleaseTerminalRequest(
                    session_id=session_id,
                    terminal_id=terminal_id,
                )
                await self.connection.release_terminal(request)
            except Exception as e:  # noqa: BLE001
                return f"Error releasing terminal: {e}"
            else:
                return f"Terminal {terminal_id} released successfully"

        return release_terminal

    def _create_run_command_with_timeout_tool(self):
        """Create a tool that runs commands with timeout support."""

        async def run_command_with_timeout(
            command: str,
            args: list[str] | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_seconds: int = 30,
            session_id: str = "default_session",
        ) -> str:
            """Execute a command with timeout support.

            Args:
                command: The command to execute
                args: Command arguments (optional)
                cwd: Working directory (optional)
                env: Environment variables (optional)
                timeout_seconds: Timeout in seconds (default: 30)
                session_id: Session ID for the request

            Returns:
                Command output or timeout/error message
            """
            try:
                # Create terminal
                create_request = CreateTerminalRequest(
                    session_id=session_id,
                    command=command,
                    args=args or [],
                    cwd=cwd,
                    env=[EnvVariable(name=k, value=v) for k, v in (env or {}).items()],
                    output_byte_limit=1048576,
                )
                create_response = await self.connection.create_terminal(create_request)
                terminal_id = create_response.terminal_id

                try:
                    # Wait for completion with timeout
                    wait_request = WaitForTerminalExitRequest(
                        session_id=session_id,
                        terminal_id=terminal_id,
                    )
                    await asyncio.wait_for(
                        self.connection.wait_for_terminal_exit(wait_request),
                        timeout=timeout_seconds,
                    )

                    # Get output
                    out_request = TerminalOutputRequest(
                        session_id=session_id,
                        terminal_id=terminal_id,
                    )
                    output_response = await self.connection.terminal_output(out_request)

                    result = output_response.output
                    if output_response.exit_status:
                        code = output_response.exit_status.exit_code
                        if code is not None:
                            result += f"\n[Command exited with code {code}]"
                        if output_response.exit_status.signal:
                            signal = output_response.exit_status.signal
                            result += f"\n[Terminated by signal {signal}]"

                except TimeoutError:
                    # Kill the command on timeout
                    try:
                        kill_request = KillTerminalCommandRequest(
                            session_id=session_id,
                            terminal_id=terminal_id,
                        )
                        await self.connection.kill_terminal(kill_request)

                        # Get partial output
                        request = TerminalOutputRequest(
                            session_id=session_id,
                            terminal_id=terminal_id,
                        )
                        output_response = await self.connection.terminal_output(request)

                        result = output_response.output
                        timeout_msg = (
                            f"Command timed out after {timeout_seconds} "
                            f"seconds and was killed"
                        )
                        result += f"\n[{timeout_msg}]"
                    except Exception:  # noqa: BLE001
                        result = (
                            f"Command timed out after {timeout_seconds} "
                            f"seconds and failed to kill"
                        )

                finally:
                    # Always release terminal
                    release_request = ReleaseTerminalRequest(
                        session_id=session_id,
                        terminal_id=terminal_id,
                    )
                    with contextlib.suppress(Exception):
                        await self.connection.release_terminal(release_request)

            except Exception as e:  # noqa: BLE001
                return f"Error executing command with timeout: {e}"

            return result

        return run_command_with_timeout

    def _register_filesystem_tools_with_agents(self) -> None:
        """Register client-side filesystem tools with all agents in the pool."""
        if not self.agent_pool or not self.agent_pool.agents:
            logger.debug("No agents in pool to register filesystem tools with")
            return

        from llmling_agent.tools.base import Tool

        # Create client-side filesystem tools
        tools = [
            Tool.from_callable(
                self._create_read_file_tool(),
                source="filesystem",
                name_override="read_text_file",
            ),
            Tool.from_callable(
                self._create_write_file_tool(),
                source="filesystem",
                name_override="write_text_file",
            ),
        ]

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

    def _create_read_file_tool(self):
        """Create a tool that reads text files via the ACP client."""

        async def read_text_file(
            path: str,
            line: int | None = None,
            limit: int | None = None,
            session_id: str = "default_session",
        ) -> str:
            """Read text file contents from the client's filesystem.

            Args:
                path: Absolute path to the file to read
                line: Optional line number to start reading from (1-based)
                limit: Optional maximum number of lines to read
                session_id: Session ID for the request

            Returns:
                File content or error message
            """
            try:
                request = ReadTextFileRequest(
                    session_id=session_id,
                    path=path,
                    line=line,
                    limit=limit,
                )
                response = await self.connection.read_text_file(request)
            except Exception as e:  # noqa: BLE001
                return f"Error reading file: {e}"
            else:
                return response.content

        return read_text_file

    def _create_write_file_tool(self):
        """Create a tool that writes text files via the ACP client."""

        async def write_text_file(
            path: str,
            content: str,
            session_id: str = "default_session",
        ) -> str:
            """Write text content to a file in the client's filesystem.

            Args:
                path: Absolute path to the file to write
                content: The text content to write to the file
                session_id: Session ID for the request

            Returns:
                Success message or error message
            """
            try:
                request = WriteTextFileRequest(
                    session_id=session_id,
                    path=path,
                    content=content,
                )
                await self.connection.write_text_file(request)
            except Exception as e:  # noqa: BLE001
                return f"Error writing file: {e}"
            else:
                return f"Successfully wrote file: {path}"

        return write_text_file
