"""Agent Interaction section of the LLMling-agent documentation."""

from __future__ import annotations

import mknodes as mk


nav = mk.MkNav("Agent Interaction")


@nav.route.page("Running Agents", icon="octicon:play-16")
def _(page: mk.MkPage):
    """Different ways to run and interact with agents."""
    page += mk.MkTemplate("docs/running_agents.md")


@nav.route.page("Multi-Agent Systems", icon="octicon:stack-16")
def _(page: mk.MkPage):
    """Working with multiple agents and orchestration patterns."""
    page += mk.MkTemplate("docs/multi_agent.md")


@nav.route.page("Decision Making", icon="octicon:git-branch-16")
def _(page: mk.MkPage):
    """Agent decision making and selection processes."""
    page += mk.MkTemplate("docs/interaction/pick.md")


@nav.route.page("Data Extraction", icon="octicon:search-16")
def _(page: mk.MkPage):
    """Extracting structured data from text."""
    page += mk.MkTemplate("docs/interaction/extract.md")


if __name__ == "__main__":
    print(nav.to_markdown())
