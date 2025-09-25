"""Agent Client protocol (ACP) implementation."""

from .core import (
    Agent,
    AgentSideConnection,
    Client,
    ClientSideConnection,
    TerminalHandle,
    create_session_model_state,
)
from .meta import (
    AGENT_METHODS,
    CLIENT_METHODS,
    PROTOCOL_VERSION,
)
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
from .stdio import stdio_streams
from .exceptions import RequestError

__version__ = "0.0.1"

__all__ = [  # noqa: RUF022
    # constants
    "PROTOCOL_VERSION",
    "AGENT_METHODS",
    "CLIENT_METHODS",
    # types
    "InitializeRequest",
    "InitializeResponse",
    "NewSessionRequest",
    "NewSessionResponse",
    "LoadSessionRequest",
    "LoadSessionResponse",
    "AuthenticateRequest",
    "AuthenticateResponse",
    "PromptRequest",
    "PromptResponse",
    "WriteTextFileRequest",
    "WriteTextFileResponse",
    "ReadTextFileRequest",
    "ReadTextFileResponse",
    "RequestPermissionRequest",
    "RequestPermissionResponse",
    "CancelNotification",
    "SessionNotification",
    "SetSessionModeRequest",
    "SetSessionModeResponse",
    # model types
    "ModelInfo",
    "SessionModelState",
    "SetSessionModelRequest",
    "SetSessionModelResponse",
    # terminal types
    "CreateTerminalRequest",
    "CreateTerminalResponse",
    "TerminalOutputRequest",
    "TerminalOutputResponse",
    "WaitForTerminalExitRequest",
    "WaitForTerminalExitResponse",
    "KillTerminalCommandRequest",
    "KillTerminalCommandResponse",
    "ReleaseTerminalRequest",
    "ReleaseTerminalResponse",
    # core
    "AgentSideConnection",
    "ClientSideConnection",
    "RequestError",
    "Agent",
    "Client",
    "TerminalHandle",
    # utilities
    "create_session_model_state",
    # stdio helper
    "stdio_streams",
]
