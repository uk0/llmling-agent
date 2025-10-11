"""MCP Registry client service for discovering and managing MCP servers.

This module provides functionality to interact with the Model Context Protocol
registry API for server discovery and configuration.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import Field, field_validator
from schemez import Schema


ServiceName = str


class RegistryRepository(Schema):
    """Repository information for a registry server."""

    url: str
    source: str
    """Repository platform (e.g., 'github')."""

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Allow empty URLs from registry."""
        return v


class RegistryTransport(Schema):
    """Transport configuration for a package."""

    type: str
    """Transport type (stdio, sse, streamable-http)."""

    url: str | None = None
    """URL for HTTP transports."""


class RegistryPackage(Schema):
    """Package information for installing an MCP server."""

    registry_type: str = Field(alias="registryType")
    """Package registry type (npm, pypi, docker)."""

    identifier: str
    """Package identifier."""

    version: str
    """Package version."""

    transport: RegistryTransport
    """Transport configuration."""

    environment_variables: list[dict[str, Any]] = Field(
        default_factory=list, alias="environmentVariables"
    )
    """Environment variables."""

    package_arguments: list[dict[str, Any]] = Field(
        default_factory=list, alias="packageArguments"
    )
    """Package arguments."""

    runtime_hint: str | None = Field(None, alias="runtimeHint")
    """Runtime hint."""

    registry_base_url: str | None = Field(None, alias="registryBaseUrl")
    """Registry base URL."""

    file_sha256: str | None = Field(None, alias="fileSha256")
    """File SHA256 hash."""


class RegistryRemote(Schema):
    """Remote endpoint configuration."""

    type: str
    """Remote type (sse, streamable-http)."""

    url: str
    """Remote URL."""

    headers: list[dict[str, Any]] = Field(default_factory=list)
    """Request headers."""


class RegistryServer(Schema):
    """MCP server entry from the registry."""

    name: ServiceName
    """Unique server identifier."""

    description: str
    """Server description."""

    version: str
    """Server version."""

    repository: RegistryRepository
    """Repository information."""

    packages: list[RegistryPackage] = Field(default_factory=list)
    """Available packages."""

    remotes: list[RegistryRemote] = Field(default_factory=list)
    """Remote endpoints."""

    schema_: str | None = Field(None, alias="$schema")
    """JSON schema URL."""


class RegistryServerWrapper(Schema):
    """Wrapper for server data from the official registry API."""

    server: RegistryServer
    """The actual server data."""

    meta: dict[str, Any] = Field(default_factory=dict, alias="_meta")
    """Registry metadata."""


class RegistryListResponse(Schema):
    """Response from the registry list servers endpoint."""

    servers: list[RegistryServerWrapper]
    """List of wrapped server entries."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Response metadata."""


class MCPRegistryClient:
    """Client for interacting with the MCP registry API."""

    def __init__(self, base_url: str = "https://registry.modelcontextprotocol.io"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)

    async def list_servers(
        self, search: str | None = None, status: str = "active"
    ) -> list[RegistryServer]:
        """List servers from registry with optional filtering."""
        try:
            response = await self.client.get(f"{self.base_url}/v0/servers")
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            msg = f"Failed to list servers: {e}"
            raise MCPRegistryError(msg) from e
        else:
            response_data = RegistryListResponse(**data)
            wrappers = response_data.servers

            if status:  # Filter by status from metadata
                wrappers = [
                    wrapper
                    for wrapper in wrappers
                    if wrapper.meta.get(
                        "io.modelcontextprotocol.registry/official", {}
                    ).get("status")
                    == status
                ]

            servers = [wrapper.server for wrapper in wrappers]
            if search:  # Filter by search term
                search_lower = search.lower()
                servers = [
                    s
                    for s in servers
                    if search_lower in s.name.lower()
                    or search_lower in s.description.lower()
                ]

            return servers

    async def get_server(self, server_id: str) -> RegistryServer:
        """Get full server details including packages."""
        try:
            # Get all wrappers to access metadata
            response = await self.client.get(f"{self.base_url}/v0/servers")
            response.raise_for_status()
            data = response.json()
            response_data = RegistryListResponse(**data)

            # Find server by name
            target_wrapper = None
            for wrapper in response_data.servers:
                if wrapper.server.name == server_id:
                    target_wrapper = wrapper
                    break

            if not target_wrapper:
                msg = f"Server {server_id!r} not found in registry"
                raise MCPRegistryError(msg)

            # Get the UUID from metadata
            server_uuid = target_wrapper.meta.get(
                "io.modelcontextprotocol.registry/official", {}
            ).get("id")

            if not server_uuid:
                msg = f"No UUID found for server {server_id!r}"
                raise MCPRegistryError(msg)

            # Now fetch the full server details using UUID
            response = await self.client.get(f"{self.base_url}/v0/servers/{server_uuid}")
            response.raise_for_status()

            server_data = response.json()
            return RegistryServer(**server_data)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:  # noqa: PLR2004
                msg = f"Server {server_id!r} not found in registry"
                raise MCPRegistryError(msg) from e
            msg = f"Failed to get server details: {e}"
            raise MCPRegistryError(msg) from e
        except httpx.HTTPError as e:
            msg = f"Failed to get server details: {e}"
            raise MCPRegistryError(msg) from e

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class MCPRegistryError(Exception):
    """Exception raised for MCP registry operations."""


if __name__ == "__main__":
    import asyncio

    import devtools

    async def main():
        """Test the MCP registry client."""
        async with MCPRegistryClient() as client:
            print("Listing servers from official registry...")
            servers = await client.list_servers()
            print(f"Found {len(servers)} servers")

            for server in servers[:3]:  # Show first 3 servers
                print(f"\n=== {server.name} ===")
                devtools.debug(server)

    asyncio.run(main())
