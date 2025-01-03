from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class MCPServerBase(BaseModel):
    """Base model for MCP server configuration."""

    type: str
    enabled: bool = True
    environment: dict[str, str] | None = None

    model_config = ConfigDict(use_attribute_docstrings=True)


class StdioMCPServer(MCPServerBase):
    """MCP server started via stdio."""

    type: Literal["stdio"] = Field("stdio", init=False)
    command: str
    args: list[str] = Field(default_factory=list)


class SSEMCPServer(MCPServerBase):
    """MCP server using Server-Sent Events transport."""

    type: Literal["sse"] = Field("sse", init=False)
    url: str


MCPServerConfig = Annotated[StdioMCPServer | SSEMCPServer, Field(discriminator="type")]
