"""Getting Started section of the LLMling-agent documentation."""

from __future__ import annotations

import mknodes as mk


nav = mk.MkNav("Getting Started")


@nav.route.page("Installation", icon="octicons:download-16")
def _(page: mk.MkPage):
    """Installation instructions."""
    page += mk.MkTemplate("docs/getting_started/installation.md")


@nav.route.page("Basic Concepts", icon="octicons:book-16")
def _(page: mk.MkPage):
    """Core concepts and architecture overview."""
    page += mk.MkTemplate("docs/getting_started/basic_concepts.md")


@nav.route.page("Quickstart", icon="octicons:rocket-16")
def _(page: mk.MkPage):
    """Quick introduction to using LLMling-agent."""
    page += mk.MkTemplate("docs/getting_started/quickstart.md")


@nav.route.page("Creating Agents", icon="octicons:tools-16")
def _(page: mk.MkPage):
    """Guide to creating and configuring agents."""
    page += mk.MkTemplate("docs/agents/create_agents.md")


if __name__ == "__main__":
    print(nav.to_markdown())
