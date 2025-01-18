"""Interaction Patterns section of the LLMling-agent documentation."""

from __future__ import annotations

import mknodes as mk


nav = mk.MkNav("Interaction Patterns")


# @nav.route.page("Command System", icon="octicon:terminal-16")
# def _(page: mk.MkPage):
#     """Built-in command system and extensibility."""
#     page += mk.MkTemplate("docs/interaction/commands.md")


# @nav.route.page("Interactive Sessions", icon="octicon:comment-discussion-16")
# def _(page: mk.MkPage):
#     """Managing interactive chat sessions."""
#     page += mk.MkTemplate("docs/interaction/sessions.md")


# @nav.route.page("Human Provider", icon="octicon:person-16")
# def _(page: mk.MkPage):
#     """Using the human provider for interaction."""
#     page += mk.MkTemplate("docs/interaction/human_provider.md")


# @nav.route.page("Input Models", icon="octicon:keyboard-16")
# def _(page: mk.MkPage):
#     """Working with input model providers."""
#     page += mk.MkTemplate("docs/interaction/input_models.md")


if __name__ == "__main__":
    print(nav.to_markdown())
