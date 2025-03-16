"""Functions to discover available MCP servers."""

from __future__ import annotations

import anyenv
from pydantic import BaseModel, Field


class ServerInfo(BaseModel):
    """Information about an MCP server."""

    name: str
    url: str
    external_url: str | None = None
    short_description: str
    source_code_url: str | None = None
    github_stars: int | None = Field(default=None)
    package_registry: str | None = None
    package_name: str | None = None
    package_download_count: int | None = Field(default=None)
    ai_generated_description: str | None = Field(
        default=None, alias="EXPERIMENTAL_ai_generated_description"
    )


class ServerListResponse(BaseModel):
    """Response from the MCP server list endpoint."""

    servers: list[ServerInfo]
    next: str | None = None
    total_count: int


async def get_mcp_servers() -> list[ServerInfo]:
    """Fetch all available MCP servers.

    Returns:
        List of ServerInfo objects representing available MCP servers.

    Raises:
        HTTPError: If the API request fails.
    """
    result = await anyenv.get_json(
        "https://api.pulsemcp.com/v0beta/servers",
        headers={"User-Agent": "MCPToolDiscovery/1.0"},
        params={"count_per_page": 5000, "offset": 0},
        return_type=ServerListResponse,
    )
    return result.servers


if __name__ == "__main__":
    import asyncio

    servers = asyncio.run(get_mcp_servers())
    print(f"Found {len(servers)} MCP servers")
    for server in servers:
        print(f"- {server.name}: {server.short_description}")
