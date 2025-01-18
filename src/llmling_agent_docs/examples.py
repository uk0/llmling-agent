"""Examples section of the LLMling-agent documentation."""

from __future__ import annotations

import mknodes as mk


nav = mk.MkNav("Examples")


# @nav.route.page("Basic Agent Creation", icon="octicon:tools-16")
# def _(page: mk.MkPage):
#     """Demonstrate creating agents from minimal configuration."""
#     page += mk.MkTemplate("docs/examples/minimal.md")


# @nav.route.page("Structured Responses", icon="octicon:project-16")
# def _(page: mk.MkPage):
#     """Using structured response types."""
#     page += mk.MkTemplate("docs/examples/structured_response.md")


# @nav.route.page("Smart Support Router", icon="octicon:git-branch-16")
# def _(page: mk.MkPage):
#     """Type-safe decision making and routing."""
#     page += mk.MkTemplate("docs/examples/structured_decisions.md")


# @nav.route.page("Pytest-Style Functions", icon="octicon:code-16")
# def _(page: mk.MkPage):
#     """Using agents with pytest-style fixtures."""
#     page += mk.MkTemplate("docs/examples/pytest_style.md")


# @nav.route.page("Office Gossip", icon="octicon:comment-discussion-16")
# def _(page: mk.MkPage):
#     """Agent communication patterns."""
#     page += mk.MkTemplate("docs/examples/gossip_talk.md")


# @nav.route.page("Download Workers", icon="octicon:download-16")
# def _(page: mk.MkPage):
#     """Using agents as tools for downloads."""
#     page += mk.MkTemplate("docs/examples/download_workers.md")


# @nav.route.page("Download Agents", icon="octicon:cloud-download-16")
# def _(page: mk.MkPage):
#     """Sequential vs parallel downloads."""
#     page += mk.MkTemplate("docs/examples/download_agents.md")


# @nav.route.page("Creating Documentation", icon="octicon:book-16")
# def _(page: mk.MkPage):
#     """Agents creating documentation."""
#     page += mk.MkTemplate("docs/examples/create_docs.md")


# @nav.route.page("Human Interaction", icon="octicon:person-16")
# def _(page: mk.MkPage):
#     """AI-Human interaction patterns."""
#     page += mk.MkTemplate("docs/examples/human_interaction.md")


if __name__ == "__main__":
    print(nav.to_markdown())
