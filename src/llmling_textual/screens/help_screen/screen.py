"""Help screen. Credits to Elia (https://github.com/darrenburns/elia)."""

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Markdown


class HelpScreen(ModalScreen[None]):
    BINDINGS: ClassVar = [
        Binding("q", "app.quit", "Quit", show=False),
        Binding("escape,f1,?", "app.pop_screen()", "Close help", key_display="esc"),
    ]

    HELP_MARKDOWN = """\
Some help text.
"""

    def compose(self) -> ComposeResult:
        with Vertical(id="help-container") as vertical:
            vertical.border_title = "Elia Help"
            with VerticalScroll():
                yield Markdown(self.HELP_MARKDOWN, id="help-markdown")
            yield Markdown(
                "Use `pageup`, `pagedown`, `up`, and `down` to scroll.",
                id="help-scroll-keys-info",
            )
        yield Footer()
