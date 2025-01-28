"""Team configuration models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from llmling_agent.models.events import EventConfig  # noqa: TC001
from llmling_agent.models.forward_targets import ForwardingTarget  # noqa: TC001
from llmling_agent.models.mcp_server import (
    MCPServerBase,
    MCPServerConfig,
    StdioMCPServer,
)


class NodeConfig(BaseModel):
    """Configuration for a Node of the messaging system."""

    name: str | None = None
    """Name of the Agent / Team"""

    description: str | None = None
    """Optional description of the agent / team."""

    triggers: list[EventConfig] = Field(default_factory=list)
    """Event sources that activate this agent / team"""

    connections: list[ForwardingTarget] = Field(default_factory=list)
    """Targets to forward results to."""

    mcp_servers: list[str | MCPServerConfig] = Field(default_factory=list)
    """List of MCP server configurations:
    - str entries are converted to StdioMCPServer
    - MCPServerConfig for full server configuration
    """

    # Future extensions:
    # tools: list[str] | None = None
    # """Tools available to all team members."""

    # knowledge: Knowledge | None = None
    # """Knowledge sources shared by all team members."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        use_attribute_docstrings=True,
    )

    def get_mcp_servers(self) -> list[MCPServerConfig]:
        """Get processed MCP server configurations.

        Converts string entries to StdioMCPServer configs by splitting
        into command and arguments.

        Returns:
            List of MCPServerConfig instances

        Raises:
            ValueError: If string entry is empty
        """
        configs: list[MCPServerConfig] = []

        for server in self.mcp_servers:
            match server:
                case str():
                    parts = server.split()
                    if not parts:
                        msg = "Empty MCP server command"
                        raise ValueError(msg)

                    configs.append(StdioMCPServer(command=parts[0], args=parts[1:]))
                case MCPServerBase():
                    configs.append(server)

        return configs
