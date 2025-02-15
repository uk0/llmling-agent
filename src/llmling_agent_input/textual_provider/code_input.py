"""Textual based code input."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, TextArea


if TYPE_CHECKING:
    from textual.app import ComposeResult


class CodeInputModal(ModalScreen[str]):
    """Modal for code input."""

    DEFAULT_CSS = """
    CodeInputModal {
        align: center middle;
    }

    .modal-container {
        width: 80%;
        height: 60%;
        border: heavy $primary;
        padding: 1;
    }

    #description {
        text-align: center;
    }

    #code-area {
        height: 1fr;
        margin: 1;
    }
    """

    BINDINGS: ClassVar = [
        Binding("escape,ctrl+c", "cancel", "Cancel"),
        Binding("ctrl+enter", "submit", "Submit"),
    ]

    def __init__(
        self,
        template: str | None = None,
        language: str = "python",
        description: str | None = None,
    ):
        super().__init__()
        self.template = template
        self.language = language
        self.description = description

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-container"):
            if self.description:
                yield Label(self.description, id="description")
            yield TextArea(self.template or "", language=self.language, id="code-area")

    def action_submit(self):
        text_area = self.query_one(TextArea)
        self.dismiss(text_area.text)

    def action_cancel(self):
        self.dismiss(None)
