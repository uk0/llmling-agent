from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from .connection import Connection
from .exceptions import RequestError
from .meta import AGENT_METHODS, CLIENT_METHODS
from .schema import (
    AuthenticateRequest,
    AuthenticateResponse,
    CancelNotification,
    CreateTerminalRequest,
    CreateTerminalResponse,
    InitializeRequest,
    InitializeResponse,
    KillTerminalCommandRequest,
    KillTerminalCommandResponse,
    LoadSessionRequest,
    LoadSessionResponse,
    NewSessionRequest,
    NewSessionResponse,
    PromptRequest,
    PromptResponse,
    ReadTextFileRequest,
    ReadTextFileResponse,
    ReleaseTerminalRequest,
    ReleaseTerminalResponse,
    RequestPermissionRequest,
    RequestPermissionResponse,
    SessionModelState,
    SessionNotification,
    SetSessionModelRequest,
    SetSessionModelResponse,
    SetSessionModeRequest,
    SetSessionModeResponse,
    TerminalOutputRequest,
    TerminalOutputResponse,
    WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
    WriteTextFileRequest,
    WriteTextFileResponse,
)


if TYPE_CHECKING:
    import asyncio
    from collections.abc import Callable

    from .acp_types import MethodHandler


_AGENT_CONNECTION_ERROR = "AgentSideConnection requires asyncio StreamWriter/StreamReader"
_CLIENT_CONNECTION_ERROR = (
    "ClientSideConnection requires asyncio StreamWriter/StreamReader"
)


class NoMatch:
    """NoMatch Sentinel."""


_NO_MATCH = NoMatch()


# --- High-level Agent/Client wrappers -------------------------------------------


class Client(Protocol):
    """High-level client interface for interacting with an ACP server."""

    async def requestPermission(
        self, params: RequestPermissionRequest
    ) -> RequestPermissionResponse: ...

    async def sessionUpdate(self, params: SessionNotification) -> None: ...

    async def writeTextFile(
        self, params: WriteTextFileRequest
    ) -> WriteTextFileResponse | None: ...

    async def readTextFile(self, params: ReadTextFileRequest) -> ReadTextFileResponse: ...

    # Optional/unstable terminal methods
    async def createTerminal(
        self, params: CreateTerminalRequest
    ) -> CreateTerminalResponse: ...

    async def terminalOutput(
        self, params: TerminalOutputRequest
    ) -> TerminalOutputResponse: ...

    async def releaseTerminal(
        self, params: ReleaseTerminalRequest
    ) -> ReleaseTerminalResponse | None: ...

    async def waitForTerminalExit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse: ...

    async def killTerminal(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse | None: ...

    # Extension hooks (optional)
    async def extMethod(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...

    async def extNotification(self, method: str, params: dict[str, Any]) -> None: ...


class Agent(Protocol):
    """ACP Agent interface."""

    async def initialize(self, params: InitializeRequest) -> InitializeResponse: ...

    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse: ...

    async def loadSession(self, params: LoadSessionRequest) -> LoadSessionResponse: ...

    async def authenticate(
        self, params: AuthenticateRequest
    ) -> AuthenticateResponse | None: ...

    async def prompt(self, params: PromptRequest) -> PromptResponse: ...

    async def cancel(self, params: CancelNotification) -> None: ...

    async def setSessionMode(
        self, params: SetSessionModeRequest
    ) -> SetSessionModeResponse | None: ...

    async def setSessionModel(
        self, params: SetSessionModelRequest
    ) -> SetSessionModelResponse | None: ...

    # Extension hooks (optional)
    async def extMethod(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...

    async def extNotification(self, method: str, params: dict[str, Any]) -> None: ...


class AgentSideConnection:
    """Agent-side connection.

    Use when you implement the Agent and need to talk to a Client.

    Args:
        to_agent: factory that receives this connection and returns your Agent
        input: asyncio.StreamWriter (local -> peer)
        output: asyncio.StreamReader (peer -> local)
    """

    def __init__(
        self,
        to_agent: Callable[[AgentSideConnection], Agent],
        input_stream: asyncio.StreamWriter,
        output_stream: asyncio.StreamReader,
    ) -> None:
        agent = to_agent(self)
        handler = _create_agent_handler(agent)
        self._conn = Connection(handler, input_stream, output_stream)

    # client-bound methods (agent -> client)
    async def sessionUpdate(self, params: SessionNotification) -> None:
        dct = params.model_dump(by_alias=True, exclude_none=True)
        await self._conn.send_notification(CLIENT_METHODS["session_update"], dct)

    async def requestPermission(
        self, params: RequestPermissionRequest
    ) -> RequestPermissionResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        method = CLIENT_METHODS["session_request_permission"]
        resp = await self._conn.send_request(method, dct)
        return RequestPermissionResponse.model_validate(resp)

    async def readTextFile(self, params: ReadTextFileRequest) -> ReadTextFileResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(CLIENT_METHODS["fs_read_text_file"], dct)
        return ReadTextFileResponse.model_validate(resp)

    async def writeTextFile(
        self, params: WriteTextFileRequest
    ) -> WriteTextFileResponse | None:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        r = await self._conn.send_request(CLIENT_METHODS["fs_write_text_file"], dct)
        # Response may be empty object
        return WriteTextFileResponse.model_validate(r) if isinstance(r, dict) else None

    # async def createTerminal(self, params: CreateTerminalRequest) -> TerminalHandle:
    #     resp = await self._conn.send_request(
    #         CLIENT_METHODS["terminal_create"],
    #         params.model_dump(exclude_none=True, exclude_defaults=True),
    #     )
    #     create_resp = CreateTerminalResponse.model_validate(resp)
    #     return TerminalHandle(create_resp.terminal_id, params.session_id, self._conn)

    async def createTerminal(
        self, params: CreateTerminalRequest
    ) -> CreateTerminalResponse:
        return CreateTerminalResponse(terminal_id="0")

    async def extMethod(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self._conn.send_request(f"_{method}", params)

    async def extNotification(self, method: str, params: dict[str, Any]) -> None:
        await self._conn.send_notification(f"_{method}", params)

    async def terminalOutput(
        self, params: TerminalOutputRequest
    ) -> TerminalOutputResponse:
        return TerminalOutputResponse(output="", truncated=False)

    async def releaseTerminal(
        self, params: ReleaseTerminalRequest
    ) -> ReleaseTerminalResponse | None:
        pass

    async def waitForTerminalExit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse:
        return WaitForTerminalExitResponse()

    async def killTerminal(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse | None:
        return KillTerminalCommandResponse()


class ClientSideConnection:
    """Client-side connection.

    Use when you implement the Client and need to talk to an Agent.

    Args:
      to_client: factory that receives this connection and returns your Client
      input: asyncio.StreamWriter (local -> peer)
      output: asyncio.StreamReader (peer -> local)
    """

    def __init__(
        self,
        to_client: Callable[[Agent], Client],
        input_stream: asyncio.StreamWriter,
        output_stream: asyncio.StreamReader,
    ) -> None:
        # Build client first so handler can delegate
        client = to_client(self)
        handler = self._create_handler(client)
        self._conn = Connection(handler, input_stream, output_stream)

    def _create_handler(self, client: Client) -> MethodHandler:
        """Create the method handler for client-side connection."""

        async def handler(
            method: str, params: Any, is_notification: bool
        ) -> (
            WriteTextFileResponse
            | ReadTextFileResponse
            | RequestPermissionResponse
            | SessionNotification
            | CreateTerminalResponse
            | TerminalOutputResponse
            | WaitForTerminalExitResponse
            | dict[str, Any]
            | NoMatch
            | None
        ):
            return await _handle_client_method(client, method, params, is_notification)

        return handler

    # agent-bound methods (client -> agent)
    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(AGENT_METHODS["initialize"], dct)
        return InitializeResponse.model_validate(resp)

    async def newSession(self, params: NewSessionRequest) -> NewSessionResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(AGENT_METHODS["session_new"], dct)
        return NewSessionResponse.model_validate(resp)

    async def loadSession(self, params: LoadSessionRequest) -> LoadSessionResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(AGENT_METHODS["session_load"], dct)
        return LoadSessionResponse.model_validate(resp)

    async def setSessionMode(
        self, params: SetSessionModeRequest
    ) -> SetSessionModeResponse | None:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        r = await self._conn.send_request(AGENT_METHODS["session_set_mode"], dct)
        # May be empty object
        return SetSessionModeResponse.model_validate(r) if isinstance(r, dict) else None

    async def setSessionModel(
        self, params: SetSessionModelRequest
    ) -> SetSessionModelResponse | None:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        r = await self._conn.send_request(AGENT_METHODS["model_select"], dct)
        # May be empty object
        return SetSessionModelResponse.model_validate(r) if isinstance(r, dict) else None

    async def authenticate(
        self, params: AuthenticateRequest
    ) -> AuthenticateResponse | None:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        r = await self._conn.send_request(AGENT_METHODS["authenticate"], dct)
        return AuthenticateResponse.model_validate(r) if isinstance(r, dict) else None

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(AGENT_METHODS["session_prompt"], dct)
        return PromptResponse.model_validate(resp)

    async def cancel(self, params: CancelNotification) -> None:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        await self._conn.send_notification(AGENT_METHODS["session_cancel"], dct)

    async def extMethod(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self._conn.send_request(f"_{method}", params)

    async def extNotification(self, method: str, params: dict[str, Any]) -> None:
        await self._conn.send_notification(f"_{method}", params)


class TerminalHandle:
    """Handle for a terminal session."""

    def __init__(self, terminal_id: str, session_id: str, conn: Connection) -> None:
        self.id = terminal_id
        self._session_id = session_id
        self._conn = conn

    async def current_output(self) -> TerminalOutputResponse:
        dct = {"sessionId": self._session_id, "terminalId": self.id}
        resp = await self._conn.send_request(CLIENT_METHODS["terminal_output"], dct)
        return TerminalOutputResponse.model_validate(resp)

    async def wait_for_exit(self) -> WaitForTerminalExitResponse:
        dct = {"sessionId": self._session_id, "terminalId": self.id}
        method = CLIENT_METHODS["terminal_wait_for_exit"]
        resp = await self._conn.send_request(method, dct)
        return WaitForTerminalExitResponse.model_validate(resp)

    async def kill(self) -> KillTerminalCommandResponse | None:
        dct = {"sessionId": self._session_id, "terminalId": self.id}
        resp = await self._conn.send_request(CLIENT_METHODS["terminal_kill"], dct)
        return (
            KillTerminalCommandResponse.model_validate(resp)
            if isinstance(resp, dict)
            else None
        )

    async def release(self) -> ReleaseTerminalResponse | None:
        dct = {"sessionId": self._session_id, "terminalId": self.id}
        resp = await self._conn.send_request(CLIENT_METHODS["terminal_release"], dct)
        return (
            ReleaseTerminalResponse.model_validate(resp)
            if isinstance(resp, dict)
            else None
        )


async def _handle_client_core_methods(
    client: Client,
    method: str,
    params: Any,
) -> (
    WriteTextFileResponse
    | ReadTextFileResponse
    | RequestPermissionResponse
    | SessionNotification
    | NoMatch
    | None
):
    if method == CLIENT_METHODS["fs_write_text_file"]:
        write_file_request = WriteTextFileRequest.model_validate(params)
        return await client.writeTextFile(write_file_request)
    if method == CLIENT_METHODS["fs_read_text_file"]:
        read_file_request = ReadTextFileRequest.model_validate(params)
        return await client.readTextFile(read_file_request)
    if method == CLIENT_METHODS["session_request_permission"]:
        permission_request = RequestPermissionRequest.model_validate(params)
        return await client.requestPermission(permission_request)
    if method == CLIENT_METHODS["session_update"]:
        notification = SessionNotification.model_validate(params)
        await client.sessionUpdate(notification)
        return None
    return _NO_MATCH


async def _handle_client_terminal_basic(
    client: Client, method: str, params: Any
) -> CreateTerminalResponse | TerminalOutputResponse | None | NoMatch:
    if method == CLIENT_METHODS["terminal_create"]:
        create_request = CreateTerminalRequest.model_validate(params)
        return await client.createTerminal(create_request)
    if method == CLIENT_METHODS["terminal_output"]:
        output_request = TerminalOutputRequest.model_validate(params)
        return await client.terminalOutput(output_request)
    return _NO_MATCH


async def _handle_client_terminal_lifecycle(
    client: Client, method: str, params: Any
) -> dict[str, Any] | WaitForTerminalExitResponse | NoMatch | None:
    if method == CLIENT_METHODS["terminal_release"]:
        release_request = ReleaseTerminalRequest.model_validate(params)
        return (
            result.model_dump(by_alias=True, exclude_none=True)
            if (result := await client.releaseTerminal(release_request))
            else {}
        )
    if method == CLIENT_METHODS["terminal_wait_for_exit"]:
        wait_request = WaitForTerminalExitRequest.model_validate(params)
        return await client.waitForTerminalExit(wait_request)
    if method == CLIENT_METHODS["terminal_kill"]:
        kill_request = KillTerminalCommandRequest.model_validate(params)
        return (
            kill_result.model_dump(by_alias=True, exclude_none=True)
            if (kill_result := await client.killTerminal(kill_request))
            else {}
        )
    return _NO_MATCH


async def _handle_client_extension_methods(
    client: Client, method: str, params: Any, is_notification: bool
) -> NoMatch | dict[str, Any] | None:
    if isinstance(method, str) and method.startswith("_"):
        ext_name = method[1:]
        if is_notification:
            await client.extNotification(ext_name, params or {})
            return None
        return await client.extMethod(ext_name, params or {})
    return _NO_MATCH


async def _handle_client_terminal_methods(
    client: Client, method: str, params: Any
) -> (
    WaitForTerminalExitResponse
    | CreateTerminalResponse
    | TerminalOutputResponse
    | None
    | dict[str, Any]
    | NoMatch
):
    if (
        result := await _handle_client_terminal_basic(client, method, params)
    ) is not _NO_MATCH:
        return result
    if (
        term_result := await _handle_client_terminal_lifecycle(client, method, params)
    ) is not _NO_MATCH:
        return term_result
    return _NO_MATCH


async def _handle_client_method(
    client: Client, method: str, params: Any, is_notification: bool
) -> (
    WriteTextFileResponse
    | ReadTextFileResponse
    | RequestPermissionResponse
    | SessionNotification
    | CreateTerminalResponse
    | TerminalOutputResponse
    | WaitForTerminalExitResponse
    | dict[str, Any]
    | NoMatch
    | None
):
    """Handle client method calls."""
    # Core session/file methods
    if (
        result := await _handle_client_core_methods(client, method, params)
    ) is not _NO_MATCH:
        return result
    # Terminal methods
    if (
        term_result := await _handle_client_terminal_methods(client, method, params)
    ) is not _NO_MATCH:
        return term_result
    # Extension methods/notifications
    if (
        ext_result := await _handle_client_extension_methods(
            client, method, params, is_notification
        )
    ) is not _NO_MATCH:
        return ext_result
    raise RequestError.method_not_found(method)


# agent


async def _handle_agent_init_methods(
    agent: Agent, method: str, params: Any
) -> NewSessionResponse | InitializeResponse | NoMatch:
    if method == AGENT_METHODS["initialize"]:
        initialize_request = InitializeRequest.model_validate(params)
        return await agent.initialize(initialize_request)
    if method == AGENT_METHODS["session_new"]:
        new_session_request = NewSessionRequest.model_validate(params)
        return await agent.newSession(new_session_request)
    return _NO_MATCH


async def _handle_agent_session_methods(
    agent: Agent, method: str, params: Any
) -> None | dict[str, Any] | PromptResponse | NoMatch:
    if method == AGENT_METHODS["session_load"]:
        load_request = LoadSessionRequest.model_validate(params)
        await agent.loadSession(load_request)
        return None
    if method == AGENT_METHODS["session_set_mode"]:
        set_mode_request = SetSessionModeRequest.model_validate(params)
        return (
            result.model_dump(by_alias=True, exclude_none=True)
            if (result := await agent.setSessionMode(set_mode_request))
            else {}
        )
    if method == AGENT_METHODS["session_prompt"]:
        prompt_request = PromptRequest.model_validate(params)
        return await agent.prompt(prompt_request)
    if method == AGENT_METHODS["session_cancel"]:
        cancel_notification = CancelNotification.model_validate(params)
        await agent.cancel(cancel_notification)
        return None
    if method == AGENT_METHODS["model_select"]:
        set_model_request = SetSessionModelRequest.model_validate(params)
        return (
            set_model_response.model_dump(by_alias=True, exclude_none=True)
            if (set_model_response := await agent.setSessionModel(set_model_request))
            else {}
        )
    return _NO_MATCH


async def _handle_agent_auth_methods(
    agent: Agent, method: str, params: Any
) -> dict[str, Any] | NoMatch:
    if method == AGENT_METHODS["authenticate"]:
        p = AuthenticateRequest.model_validate(params)
        result = await agent.authenticate(p)
        return result.model_dump(by_alias=True, exclude_none=True) if result else {}
    return _NO_MATCH


async def _handle_agent_ext_methods(
    agent: Agent, method: str, params: Any, is_notification: bool
) -> dict[str, Any] | NoMatch | None:
    if isinstance(method, str) and method.startswith("_"):
        ext_name = method[1:]
        if is_notification:
            await agent.extNotification(ext_name, params or {})
            return None
        return await agent.extMethod(ext_name, params or {})
    return _NO_MATCH


async def _handle_agent_method(
    agent: Agent, method: str, params: Any, is_notification: bool
) -> (
    NewSessionResponse
    | InitializeResponse
    | PromptResponse
    | dict[str, Any]
    | NoMatch
    | None
):
    # Init/new
    if (
        init_result := await _handle_agent_init_methods(agent, method, params)
    ) is not _NO_MATCH:
        return init_result
    # Session-related
    if (
        result := await _handle_agent_session_methods(agent, method, params)
    ) is not _NO_MATCH:
        return result
    # Auth
    if (
        auth_result := await _handle_agent_auth_methods(agent, method, params)
    ) is not _NO_MATCH:
        return auth_result
    # Extensions
    if (
        ext_result := await _handle_agent_ext_methods(
            agent, method, params, is_notification
        )
    ) is not _NO_MATCH:
        return ext_result
    raise RequestError.method_not_found(method)


def _create_agent_handler(agent: Agent) -> MethodHandler:
    async def handler(method: str, params: Any, is_notification: bool) -> Any:
        return await _handle_agent_method(agent, method, params, is_notification)

    return handler


def create_session_model_state(
    available_models: list[str], current_model: str | None = None
) -> SessionModelState | None:
    """Create a SessionModelState from available models.

    Args:
        available_models: List of all models the agent can switch between
        current_model: The currently active model (defaults to first available)

    Returns:
        SessionModelState with all available models, None if no models provided
    """
    if not available_models:
        return None

    # Import here to avoid circular imports
    from .schema import ModelInfo

    # Create ModelInfo objects for each available model
    model_infos = []
    for model_id in available_models:
        # Extract display name (e.g., "gpt-4" from "openai:gpt-4")
        display_name = model_id.split(":")[-1] if ":" in model_id else model_id
        info = ModelInfo(model_id=model_id, name=display_name, description=model_id)
        model_infos.append(info)

    # Use first model as current if not specified
    current_model_id = current_model or available_models[0]

    return SessionModelState(
        available_models=model_infos, current_model_id=current_model_id
    )
