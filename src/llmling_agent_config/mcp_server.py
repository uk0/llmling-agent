"""MCP server configuration."""

from __future__ import annotations

import os
from typing import Annotated, Literal

from pydantic import Field
from schemez import Schema


class MCPServerAuthSettings(Schema):
    """Represents authentication configuration for a server.

    Minimal OAuth v2.1 support with sensible defaults.
    """

    oauth: bool = False

    # Local callback server configuration
    redirect_port: int = 3030
    redirect_path: str = "/callback"

    # Optional scope override. If set to a list, values are space-joined.
    scope: str | list[str] | None = None

    # Token persistence: use OS keychain via 'keyring' by default; fallback to 'memory'.
    persist: Literal["keyring", "memory"] = "keyring"


class BaseMCPServerConfig(Schema):
    """Base model for MCP server configuration."""

    type: str
    """Type discriminator for MCP server configurations."""

    name: str | None = None
    """Optional name for referencing the server."""

    enabled: bool = True
    """Whether this server is currently enabled."""

    env: dict[str, str] | None = None
    """Environment variables to pass to the server process."""

    timeout: float = 30.0
    """Timeout for the server process."""

    def get_env_vars(self) -> dict[str, str]:
        """Get environment variables for the server process."""
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        env["PYTHONIOENCODING"] = "utf-8"
        return env


class StdioMCPServerConfig(BaseMCPServerConfig):
    """MCP server started via stdio.

    Uses subprocess communication through standard input/output streams.
    """

    type: Literal["stdio"] = Field("stdio", init=False)
    """Stdio server coniguration."""

    command: str
    """Command to execute (e.g. "pipx", "python", "node")."""

    args: list[str] = Field(default_factory=list)
    """Command arguments (e.g. ["run", "some-server", "--debug"])."""

    @classmethod
    def from_string(cls, command: str) -> StdioMCPServerConfig:
        """Create a MCP server from a command string."""
        cmd, args = command.split(maxsplit=1)
        return cls(command=cmd, args=args.split())


class SSEMCPServerConfig(BaseMCPServerConfig):
    """MCP server using Server-Sent Events transport.

    Connects to a server over HTTP with SSE for real-time communication.
    """

    type: Literal["sse"] = Field("sse", init=False)
    """SSE server configuration."""

    url: str
    """URL of the SSE server endpoint."""

    auth: MCPServerAuthSettings = Field(default_factory=MCPServerAuthSettings)
    """OAuth settings for the SSE server."""


class StreamableHTTPMCPServerConfig(BaseMCPServerConfig):
    """MCP server using StreamableHttp.

    Connects to a server over HTTP with streamable HTTP.
    """

    type: Literal["streamable-http"] = Field("streamable-http", init=False)
    """HTTP server configuration."""

    url: str
    """URL of the HTTP server endpoint."""

    auth: MCPServerAuthSettings = Field(default_factory=MCPServerAuthSettings)
    """OAuth settings for the HTTP server."""


MCPServerConfig = Annotated[
    StdioMCPServerConfig | SSEMCPServerConfig | StreamableHTTPMCPServerConfig,
    Field(discriminator="type"),
]
