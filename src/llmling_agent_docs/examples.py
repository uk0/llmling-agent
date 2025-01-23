"""Examples section of the LLMling-agent documentation."""

from __future__ import annotations

import pathlib

import mknodes as mk

from llmling_agent.agent.agent import Agent


nav = mk.MkNav("Examples")

INTRO = """
"This page is generated using another older work of mine, an
experimental MkDocs add-on (or almost fork) named **MkNodes**,
which focusses on programmatic website generation. You can see the source
for this homepage section below.  This system is perfectly suited for Agent usage for
documentation generation.
"""


@nav.route.page("Creating Documentation", icon="octicon:book-16", hide="toc")
def _(page: mk.MkPage):
    """Agents creating documentation."""
    page += mk.MkTemplate("docs/examples/create_docs.md")


@nav.route.page("Download Agents", icon="octicon:download-16", hide="toc")
def _(page: mk.MkPage):
    """Sequential vs parallel downloads with cheerleader."""
    page += mk.MkTemplate("docs/examples/download_agents.md")


@nav.route.page("Pytest-Style Functions", icon="octicon:code-16", hide="toc")
def _(page: mk.MkPage):
    """Using agents with pytest-style fixtures."""
    page += mk.MkTemplate("docs/examples/pytest_style.md")


@nav.route.page("Human Interaction", icon="octicon:person-16", hide="toc")
def _(page: mk.MkPage):
    """AI-Human interaction patterns."""
    page += mk.MkTemplate("docs/examples/human_interaction.md")


@nav.route.page("MCP Servers", icon="octicon:server-16", hide="toc")
def _(page: mk.MkPage):
    """MCP server usage."""
    page += mk.MkTemplate("docs/examples/mcp_servers.md")


@nav.route.page("Expert Selection", icon="octicon:people-16", hide="toc")
def _(page: mk.MkPage):
    """Type-safe expert selection with pick()."""
    page += mk.MkTemplate("docs/examples/pick_team.md")


@nav.route.page("Structured Responses", icon="simple-icons:instructure", hide="toc")
def _(page: mk.MkPage):
    """Using structured response types."""
    page += mk.MkTemplate("docs/examples/structured_response.md")


@nav.route.page(
    "Round-robin communication YAML Edition", icon="octicon:project-16", hide="toc"
)
def _(page: mk.MkPage):
    """Using structured response types."""
    page += mk.MkTemplate("docs/examples/round_robin.md")


@nav.route.page(
    "MkDocs Integration & Docs generation", icon="oui:documentation", hide="toc"
)
def gen_docs(page: mk.MkPage):
    """Generate docs using agents."""
    agent = Agent[None](model="openai:gpt-4o-mini")
    content = pathlib.Path("src/llmling_agent/__init__.py")
    page += mk.MkAdmonition(INTRO)
    page += mk.MkCode(pathlib.Path(__file__).read_text())
    result = agent.run_sync(
        "Group and list the given classes. Use markdown for the group headers", content
    )
    page += result.content


if __name__ == "__main__":
    print(nav.to_markdown())
