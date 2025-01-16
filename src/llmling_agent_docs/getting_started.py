"""Getting Started section of the LLMling-agent documentation."""

from __future__ import annotations

import mknodes as mk


nav = mk.MkNav("Getting Started")


@nav.route.page("Welcome to LLMling-Agent", hide=["toc", "nav"], is_homepage=True)
def _(page: mk.MkPage):
    page += mk.MkTemplate("docs/home.md")


@nav.route.page("Installation", icon="octicon:download-16")
def _(page: mk.MkPage):
    """Installation instructions."""
    page += mk.MkTemplate("docs/getting_started/installation.md")


@nav.route.page("Basic Concepts", icon="octicon:book-16")
def _(page: mk.MkPage):
    """Core concepts and architecture overview."""
    page += mk.MkTemplate("docs/getting_started/basic_concepts.md")


@nav.route.page("Quickstart", icon="octicon:rocket-16")
def _(page: mk.MkPage):
    """Quick introduction to using LLMling-agent."""
    page += mk.MkTemplate("docs/getting_started/quickstart.md")


@nav.route.page("Creating Agents", icon="octicon:tools-16")
def _(page: mk.MkPage):
    """Guide to creating and configuring agents."""
    page += mk.MkTemplate("docs/agents/create_agents.md")


if __name__ == "__main__":
    print(nav.to_markdown())
