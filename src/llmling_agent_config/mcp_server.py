"""MCP server configuration."""

from __future__ import annotations

import os
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseMCPServerConfig(BaseModel):
    """Base model for MCP server configuration."""

    type: str
    """Type discriminator for MCP server configurations."""

    name: str | None = None
    """Optional name for referencing the server."""

    enabled: bool = True
    """Whether this server is currently enabled."""

    environment: dict[str, str] | None = None
    """Environment variables to pass to the server process."""

    timeout: float = 30.0
    """Timeout for the server process."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    def get_env_vars(self) -> dict[str, str]:
        """Get environment variables for the server process."""
        env = os.environ.copy()
        if self.environment:
            env.update(self.environment)
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


MCPServerConfig = Annotated[
    StdioMCPServerConfig | SSEMCPServerConfig, Field(discriminator="type")
]


class PoolServerConfig(BaseModel):
    """Configuration for pool-based MCP server."""

    enabled: bool = False
    """Whether this server is currently enabled."""

    # Resource exposure control
    serve_nodes: list[str] | bool = True
    """Which nodes to expose as tools:
    - True: All nodes
    - False: No nodes
    - list[str]: Specific node names
    """

    serve_prompts: list[str] | bool = True
    """Which prompts to expose:
    - True: All prompts from manifest
    - False: No prompts
    - list[str]: Specific prompt names
    """

    # Transport configuration
    transport: Literal["stdio", "sse"] = "stdio"
    """Transport type to use."""

    host: str = "localhost"
    """Host to bind server to (SSE only)."""

    port: int = 8000
    """Port to listen on (SSE only)."""

    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    """Allowed CORS origins (SSE only)."""

    zed_mode: bool = False
    """Enable Zed editor compatibility mode."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")

    def should_serve_node(self, name: str) -> bool:
        """Check if a node should be exposed."""
        match self.serve_nodes:
            case True:
                return True
            case False:
                return False
            case list():
                return name in self.serve_nodes
            case _:
                return False

    def should_serve_prompt(self, name: str) -> bool:
        """Check if a prompt should be exposed."""
        match self.serve_prompts:
            case True:
                return True
            case False:
                return False
            case list():
                return name in self.serve_prompts
            case _:
                return False
