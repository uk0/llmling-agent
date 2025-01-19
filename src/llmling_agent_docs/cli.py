from __future__ import annotations

import mknodes as mk


nav = mk.MkNav("CLI")

CLI_PATH = "llmling_agent.__main__:cli"


@nav.route.page(is_index=True, hide="toc")
def _(page: mk.MkPage):
    # page += mk.MkBinaryImage.for_file("docs/assets/cli.gif")
    # page += mk.MkTemplate("cli_index.jinja")
    page += mk.MkCliDoc(CLI_PATH)


@nav.route.page("add", icon="wrench")
def _(page: mk.MkPage):
    page += mk.MkCliDoc(CLI_PATH, prog_name="add")


@nav.route.page("run", icon="web")
def _(page: mk.MkPage):
    page += mk.MkCliDoc(CLI_PATH, prog_name="run")


@nav.route.page("list", icon="folder-wrench")
def _(page: mk.MkPage):
    page += mk.MkCliDoc(CLI_PATH, prog_name="list")


@nav.route.page("set", icon="folder-wrench")
def _(page: mk.MkPage):
    page += mk.MkCliDoc(CLI_PATH, prog_name="set")


@nav.route.page("chat", icon="folder-wrench")
def _(page: mk.MkPage):
    page += mk.MkCliDoc(CLI_PATH, prog_name="chat")


@nav.route.page("quickstart", icon="folder-wrench")
def _(page: mk.MkPage):
    page += mk.MkCliDoc(CLI_PATH, prog_name="quickstart")


@nav.route.page("task", icon="folder-wrench")
def _(page: mk.MkPage):
    page += mk.MkCliDoc(CLI_PATH, prog_name="task")


@nav.route.page("watch", icon="folder-wrench")
def _(page: mk.MkPage):
    page += mk.MkCliDoc(CLI_PATH, prog_name="watch")


@nav.route.page("history", icon="folder-wrench")
def _(page: mk.MkPage):
    page += mk.MkCliDoc(CLI_PATH, prog_name="history")


@nav.route.page("launch", icon="folder-wrench")
def _(page: mk.MkPage):
    page += mk.MkCliDoc(CLI_PATH, prog_name="launch")


@nav.route.page("web", icon="folder-wrench")
def _(page: mk.MkPage):
    page += mk.MkCliDoc(CLI_PATH, prog_name="web")
