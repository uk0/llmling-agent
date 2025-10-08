from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from acp.connection import Connection
from acp.exceptions import RequestError
from acp.meta import AGENT_METHODS, CLIENT_METHODS
from acp.schema import (
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
    ModelInfo,
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
    from collections.abc import Callable, Sequence

    from tokonomics.model_discovery.model_info import ModelInfo as TokoModelInfo

    from acp.acp_types import MethodHandler


class NoMatch:
    """NoMatch Sentinel."""


_NO_MATCH = NoMatch()


class BaseClient(Protocol):
    """Base client interface for ACP - always required."""

    async def request_permission(
        self, params: RequestPermissionRequest
    ) -> RequestPermissionResponse: ...

    async def session_update(self, params: SessionNotification) -> None: ...


class FileSystemCapability(Protocol):
    """Client capability for filesystem operations."""

    async def write_text_file(
        self, params: WriteTextFileRequest
    ) -> WriteTextFileResponse | None: ...

    async def read_text_file(
        self, params: ReadTextFileRequest
    ) -> ReadTextFileResponse: ...


class TerminalCapability(Protocol):
    """Client capability for terminal operations."""

    async def create_terminal(
        self, params: CreateTerminalRequest
    ) -> CreateTerminalResponse: ...

    async def terminal_output(
        self, params: TerminalOutputRequest
    ) -> TerminalOutputResponse: ...

    async def release_terminal(
        self, params: ReleaseTerminalRequest
    ) -> ReleaseTerminalResponse | None: ...

    async def wait_for_terminal_exit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse: ...

    async def kill_terminal(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse | None: ...


class ExtensibilityCapability(Protocol):
    """Client capability for extension methods."""

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None: ...


class Client(
    BaseClient, FileSystemCapability, TerminalCapability, ExtensibilityCapability
):
    """High-level client interface for interacting with an ACP server.

    Includes all client capabilities.
    New implementations should inherit from specific capability protocols instead.
    """


class BaseAgent(Protocol):
    """Base agent interface for ACP - always required."""

    async def initialize(self, params: InitializeRequest) -> InitializeResponse: ...

    async def new_session(self, params: NewSessionRequest) -> NewSessionResponse: ...

    async def prompt(self, params: PromptRequest) -> PromptResponse: ...

    async def cancel(self, params: CancelNotification) -> None: ...


class SessionPersistenceCapability(Protocol):
    """Agent capability for session persistence and authentication."""

    async def load_session(self, params: LoadSessionRequest) -> LoadSessionResponse: ...

    async def authenticate(
        self, params: AuthenticateRequest
    ) -> AuthenticateResponse | None: ...


class SessionModeCapability(Protocol):
    """Agent capability for session mode switching."""

    async def set_session_mode(
        self, params: SetSessionModeRequest
    ) -> SetSessionModeResponse | None: ...


class SessionModelCapability(Protocol):
    """Agent capability for session model switching."""

    async def set_session_model(
        self, params: SetSessionModelRequest
    ) -> SetSessionModelResponse | None: ...


class AgentExtensibilityCapability(Protocol):
    """Agent capability for extension methods."""

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None: ...


class Agent(
    BaseAgent,
    SessionPersistenceCapability,
    SessionModeCapability,
    SessionModelCapability,
    AgentExtensibilityCapability,
):
    """ACP Agent interface.

    Backward compatibility class that includes all agent capabilities.
    New implementations should inherit from specific capability protocols instead.
    """


class AgentSideConnection(Client):
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
    async def session_update(self, params: SessionNotification) -> None:
        dct = params.model_dump(by_alias=True, exclude_none=True)
        await self._conn.send_notification(CLIENT_METHODS["session_update"], dct)

    async def request_permission(
        self, params: RequestPermissionRequest
    ) -> RequestPermissionResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        method = CLIENT_METHODS["session_request_permission"]
        resp = await self._conn.send_request(method, dct)
        return RequestPermissionResponse.model_validate(resp)

    async def read_text_file(self, params: ReadTextFileRequest) -> ReadTextFileResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(CLIENT_METHODS["fs_read_text_file"], dct)
        return ReadTextFileResponse.model_validate(resp)

    async def write_text_file(
        self, params: WriteTextFileRequest
    ) -> WriteTextFileResponse | None:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        r = await self._conn.send_request(CLIENT_METHODS["fs_write_text_file"], dct)
        return WriteTextFileResponse.model_validate(r) if isinstance(r, dict) else None

    # async def createTerminal(self, params: CreateTerminalRequest) -> TerminalHandle:
    async def create_terminal(
        self, params: CreateTerminalRequest
    ) -> CreateTerminalResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(CLIENT_METHODS["terminal_create"], dct)
        #  resp = CreateTerminalResponse.model_validate(resp)
        #  return TerminalHandle(resp.terminal_id, params.session_id, self._conn)
        return CreateTerminalResponse.model_validate(resp)

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self._conn.send_request(f"_{method}", params)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        await self._conn.send_notification(f"_{method}", params)

    async def terminal_output(
        self, params: TerminalOutputRequest
    ) -> TerminalOutputResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(CLIENT_METHODS["terminal_output"], dct)
        return TerminalOutputResponse.model_validate(resp)

    async def release_terminal(
        self, params: ReleaseTerminalRequest
    ) -> ReleaseTerminalResponse | None:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(CLIENT_METHODS["terminal_release"], dct)
        return (
            ReleaseTerminalResponse.model_validate(resp)
            if isinstance(resp, dict)
            else None
        )

    async def wait_for_terminal_exit(
        self, params: WaitForTerminalExitRequest
    ) -> WaitForTerminalExitResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(
            CLIENT_METHODS["terminal_wait_for_exit"], dct
        )
        return WaitForTerminalExitResponse.model_validate(resp)

    async def kill_terminal(
        self, params: KillTerminalCommandRequest
    ) -> KillTerminalCommandResponse | None:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(CLIENT_METHODS["terminal_kill"], dct)
        return (
            KillTerminalCommandResponse.model_validate(resp)
            if isinstance(resp, dict)
            else None
        )


class ClientSideConnection(Agent):
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
            method: str, params: dict[str, Any] | None, is_notification: bool
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

    async def new_session(self, params: NewSessionRequest) -> NewSessionResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(AGENT_METHODS["session_new"], dct)
        return NewSessionResponse.model_validate(resp)

    async def load_session(self, params: LoadSessionRequest) -> LoadSessionResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(AGENT_METHODS["session_load"], dct)
        payload = resp if isinstance(resp, dict) else {}
        return LoadSessionResponse.model_validate(payload)

    async def set_session_mode(
        self, params: SetSessionModeRequest
    ) -> SetSessionModeResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(AGENT_METHODS["session_set_mode"], dct)
        payload = resp if isinstance(resp, dict) else {}
        return SetSessionModeResponse.model_validate(payload)

    async def set_session_model(
        self, params: SetSessionModelRequest
    ) -> SetSessionModelResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(AGENT_METHODS["session_set_model"], dct)
        payload = resp if isinstance(resp, dict) else {}
        return SetSessionModelResponse.model_validate(payload)

    async def authenticate(self, params: AuthenticateRequest) -> AuthenticateResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(AGENT_METHODS["authenticate"], dct)
        payload = resp if isinstance(resp, dict) else {}
        return AuthenticateResponse.model_validate(payload)

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        resp = await self._conn.send_request(AGENT_METHODS["session_prompt"], dct)
        return PromptResponse.model_validate(resp)

    async def cancel(self, params: CancelNotification) -> None:
        dct = params.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        await self._conn.send_notification(AGENT_METHODS["session_cancel"], dct)

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self._conn.send_request(f"_{method}", params)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        await self._conn.send_notification(f"_{method}", params)


async def _handle_client_core_methods(
    client: Client,
    method: str,
    params: dict[str, Any] | None,
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
        return await client.write_text_file(write_file_request)
    if method == CLIENT_METHODS["fs_read_text_file"]:
        read_file_request = ReadTextFileRequest.model_validate(params)
        return await client.read_text_file(read_file_request)
    if method == CLIENT_METHODS["session_request_permission"]:
        permission_request = RequestPermissionRequest.model_validate(params)
        return await client.request_permission(permission_request)
    if method == CLIENT_METHODS["session_update"]:
        notification = SessionNotification.model_validate(params)
        await client.session_update(notification)
        return None
    return _NO_MATCH


async def _handle_client_extension_methods(
    client: Client,
    method: str,
    params: dict[str, Any] | None,
    is_notification: bool,
) -> NoMatch | dict[str, Any] | None:
    if not method.startswith("_"):
        return _NO_MATCH
    ext_name = method[1:]
    if is_notification:
        await client.ext_notification(ext_name, params or {})
        return None
    return await client.ext_method(ext_name, params or {})


async def _handle_client_terminal_methods(
    client: Client,
    method: str,
    params: dict[str, Any] | None,
) -> (
    WaitForTerminalExitResponse
    | CreateTerminalResponse
    | TerminalOutputResponse
    | None
    | dict[str, Any]
    | NoMatch
):
    if method == CLIENT_METHODS["terminal_create"]:
        create_request = CreateTerminalRequest.model_validate(params)
        return await client.create_terminal(create_request)
    if method == CLIENT_METHODS["terminal_output"]:
        output_request = TerminalOutputRequest.model_validate(params)
        return await client.terminal_output(output_request)
    if method == CLIENT_METHODS["terminal_release"]:
        release_request = ReleaseTerminalRequest.model_validate(params)
        return (
            result.model_dump(by_alias=True, exclude_none=True)
            if (result := await client.release_terminal(release_request))
            else {}
        )
    if method == CLIENT_METHODS["terminal_wait_for_exit"]:
        wait_request = WaitForTerminalExitRequest.model_validate(params)
        return await client.wait_for_terminal_exit(wait_request)
    if method == CLIENT_METHODS["terminal_kill"]:
        kill_request = KillTerminalCommandRequest.model_validate(params)
        return (
            kill_result.model_dump(by_alias=True, exclude_none=True)
            if (kill_result := await client.kill_terminal(kill_request))
            else {}
        )
    return _NO_MATCH


async def _handle_client_method(
    client: Client,
    method: str,
    params: dict[str, Any] | None,
    is_notification: bool,
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
    if (
        result := await _handle_client_core_methods(client, method, params)
    ) is not _NO_MATCH:
        return result
    if (
        term_result := await _handle_client_terminal_methods(client, method, params)
    ) is not _NO_MATCH:
        return term_result
    if (
        ext_result := await _handle_client_extension_methods(
            client, method, params, is_notification
        )
    ) is not _NO_MATCH:
        return ext_result
    raise RequestError.method_not_found(method)


# agent


async def _handle_agent_init_methods(
    agent: Agent,
    method: str,
    params: dict[str, Any] | None,
) -> NewSessionResponse | InitializeResponse | NoMatch:
    if method == AGENT_METHODS["initialize"]:
        initialize_request = InitializeRequest.model_validate(params)
        return await agent.initialize(initialize_request)
    if method == AGENT_METHODS["session_new"]:
        new_session_request = NewSessionRequest.model_validate(params)
        return await agent.new_session(new_session_request)
    return _NO_MATCH


async def _handle_agent_session_methods(
    agent: Agent,
    method: str,
    params: dict[str, Any] | None,
) -> None | dict[str, Any] | PromptResponse | NoMatch:
    if method == AGENT_METHODS["session_load"]:
        load_request = LoadSessionRequest.model_validate(params)
        await agent.load_session(load_request)
        return None
    if method == AGENT_METHODS["session_set_mode"]:
        set_mode_request = SetSessionModeRequest.model_validate(params)
        return (
            session_resp.model_dump(by_alias=True, exclude_none=True)
            if (session_resp := await agent.set_session_mode(set_mode_request))
            else {}
        )
    if method == AGENT_METHODS["session_prompt"]:
        prompt_request = PromptRequest.model_validate(params)
        return await agent.prompt(prompt_request)
    if method == AGENT_METHODS["session_cancel"]:
        cancel_notification = CancelNotification.model_validate(params)
        await agent.cancel(cancel_notification)
        return None
    if method == AGENT_METHODS["session_set_model"]:
        set_model_request = SetSessionModelRequest.model_validate(params)
        return (
            model_result.model_dump(by_alias=True, exclude_none=True)
            if (model_result := await agent.set_session_model(set_model_request))
            else {}
        )
    return _NO_MATCH


async def _handle_agent_auth_methods(
    agent: Agent,
    method: str,
    params: dict[str, Any] | None,
) -> dict[str, Any] | NoMatch:
    if method == AGENT_METHODS["authenticate"]:
        p = AuthenticateRequest.model_validate(params)
        result = await agent.authenticate(p)
        return result.model_dump(by_alias=True, exclude_none=True) if result else {}
    return _NO_MATCH


async def _handle_agent_ext_methods(
    agent: Agent,
    method: str,
    params: dict[str, Any] | None,
    is_notification: bool,
) -> dict[str, Any] | NoMatch | None:
    if method.startswith("_"):
        ext_name = method[1:]
        if is_notification:
            await agent.ext_notification(ext_name, params or {})
            return None
        return await agent.ext_method(ext_name, params or {})
    return _NO_MATCH


async def _handle_agent_method(
    agent: Agent,
    method: str,
    params: dict[str, Any] | None,
    is_notification: bool,
) -> (
    NewSessionResponse
    | InitializeResponse
    | PromptResponse
    | dict[str, Any]
    | NoMatch
    | None
):
    if (
        init_result := await _handle_agent_init_methods(agent, method, params)
    ) is not _NO_MATCH:
        return init_result
    if (
        result := await _handle_agent_session_methods(agent, method, params)
    ) is not _NO_MATCH:
        return result
    if (
        auth_result := await _handle_agent_auth_methods(agent, method, params)
    ) is not _NO_MATCH:
        return auth_result
    if (
        ext_result := await _handle_agent_ext_methods(
            agent, method, params, is_notification
        )
    ) is not _NO_MATCH:
        return ext_result
    raise RequestError.method_not_found(method)


def _create_agent_handler(agent: Agent) -> MethodHandler:
    async def handler(
        method: str,
        params: dict[str, Any] | None,
        is_notification: bool,
    ) -> Any:
        return await _handle_agent_method(agent, method, params, is_notification)

    return handler


def create_session_model_state(
    available_models: Sequence[TokoModelInfo], current_model: str | None = None
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
    # Create ModelInfo objects for each available model
    models = [
        ModelInfo(
            model_id=model.pydantic_ai_id,
            name=f"{model.provider}: {model.name}",
            description=model.description,
        )
        for model in available_models
    ]

    # Use first model as current if not specified
    current_model_id = current_model or available_models[0].pydantic_ai_id
    return SessionModelState(available_models=models, current_model_id=current_model_id)
