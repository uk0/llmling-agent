"""YAML Configuration section of the LLMling-agent documentation."""

from __future__ import annotations

import mknodes as mk


nav = mk.MkNav("YAML Configuration")


@nav.route.page("Manifest Overview", icon="octicons:file-code-16")
def _(page: mk.MkPage):
    """Complete manifest structure and organization."""
    page += mk.MkTemplate("docs/config_file/manifest.md")


@nav.route.page("Environment Setup", icon="octicons:tools-16")
def _(page: mk.MkPage):
    """Environment configuration for tools and resources."""
    page += mk.MkTemplate("docs/config_file/env_config.md")


@nav.route.page("Provider Configuration", icon="octicons:plug-16")
def _(page: mk.MkPage):
    """Agent provider setup and options."""
    page += mk.MkTemplate("docs/config_file/provider_config.md")


@nav.route.page("Model Configuration", icon="octicons:cpu-16")
def _(page: mk.MkPage):
    """Language model setup and configuration."""
    page += mk.MkTemplate("docs/config_file/model_config.md")


@nav.route.page("Capabilities", icon="octicons:shield-check-16")
def _(page: mk.MkPage):
    """Agent capabilities and permissions."""
    page += mk.MkTemplate("docs/config_file/capabilities_config.md")


@nav.route.page("Worker Configuration", icon="octicons:people-16")
def _(page: mk.MkPage):
    """Worker agent setup and management."""
    page += mk.MkTemplate("docs/config_file/worker_config.md")


@nav.route.page("Response Types", icon="octicons:reply-16")
def _(page: mk.MkPage):
    """Structured response type configuration."""
    page += mk.MkTemplate("docs/config_file/response_config.md")


@nav.route.page("Knowledge Sources", icon="octicons:database-16")
def _(page: mk.MkPage):
    """Knowledge source configuration and management."""
    page += mk.MkTemplate("docs/config_file/knowledge_config.md")


@nav.route.page("Message Forwarding", icon="octicons:arrow-right-16")
def _(page: mk.MkPage):
    """Message routing and forwarding configuration."""
    page += mk.MkTemplate("docs/config_file/forward_config.md")


@nav.route.page("Storage Configuration", icon="octicons:database-16")
def _(page: mk.MkPage):
    """Database and storage setup."""
    page += mk.MkTemplate("docs/config_file/storage_config.md")


@nav.route.page("Event Configuration", icon="octicons:broadcast-16")
def _(page: mk.MkPage):
    """Event handling and trigger setup."""
    page += mk.MkTemplate("docs/config_file/events_config.md")


@nav.route.page("MCP Server Setup", icon="octicons:server-16")
def _(page: mk.MkPage):
    """MCP server configuration and integration."""
    page += mk.MkTemplate("docs/config_file/mcp_config.md")


if __name__ == "__main__":
    print(nav.to_markdown())
