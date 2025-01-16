"""Advanced Features section of the LLMling-agent documentation."""

from __future__ import annotations

import mknodes as mk


nav = mk.MkNav("Advanced Features")


@nav.route.page("Basic Agent", icon="octicon:person-16")
def _(page: mk.MkPage):
    """Core Agent implementation and features."""
    page += mk.MkTemplate("docs/agents/basic_agent.md")


@nav.route.page("Structured Agent", icon="octicon:package-16")
def _(page: mk.MkPage):
    """Type-safe structured output agents."""
    page += mk.MkTemplate("docs/agents/structured_agent.md")


@nav.route.page("Teams", icon="octicon:people-16")
def _(page: mk.MkPage):
    """Working with agent teams and groups."""
    page += mk.MkTemplate("docs/agents/team.md")


@nav.route.page("Capabilities System", icon="octicon:shield-check-16")
def _(page: mk.MkPage):
    """Agent capabilities and permissions system."""
    page += mk.MkTemplate("docs/advanced/capabilities.md")


@nav.route.page("Connections", icon="octicon:link-16")
def _(page: mk.MkPage):
    """Agent communication and connections."""
    page += mk.MkTemplate("docs/advanced/connections.md")


@nav.route.page("Agent Injection", icon="octicon:git-merge-16")
def _(page: mk.MkPage):
    """Dependency injection for agents."""
    page += mk.MkTemplate("docs/advanced/injection.md")


@nav.route.page("Event System", icon="octicon:broadcast-16")
def _(page: mk.MkPage):
    """Event handling and automation."""
    page += mk.MkTemplate("docs/advanced/events.md")


@nav.route.page("Database Integration", icon="octicon:database-16")
def _(page: mk.MkPage):
    """Database logging and storage integration."""
    page += mk.MkTemplate("docs/advanced/db.md")


if __name__ == "__main__":
    print(nav.to_markdown())
