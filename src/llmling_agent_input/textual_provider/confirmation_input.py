"""Textual based confirmation input."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label


if TYPE_CHECKING:
    from textual.app import ComposeResult


class ConfirmationModal(ModalScreen[str]):
    """Modal for tool confirmation."""

    DEFAULT_CSS = """
    ConfirmationModal {
        align: center middle;
    }

    .modal-container {
        width: 60%;
        height: auto;
        border: heavy $primary;
        padding: 1;
    }
    """

    def __init__(self, prompt: str):
        super().__init__()
        self.prompt = prompt

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-container"):
            yield Label(self.prompt)
            with Horizontal(classes="buttons"):
                yield Button("Yes", variant="success", id="yes")
                yield Button("No", variant="primary", id="no")
                yield Button("Abort", variant="warning", id="abort")
                yield Button("Quit", variant="error", id="quit")

    def on_button_pressed(self, event: Button.Pressed):
        match event.button.id:
            case "yes":
                self.dismiss("allow")
            case "abort":
                self.dismiss("abort_run")
            case "quit":
                self.dismiss("abort_chain")
            case _:
                self.dismiss("skip")
