"""Pool selection screen."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.config.store import ConfigStore
from textual.containers import Horizontal
from textual.screen import ModalScreen

from llmling_agent.log import get_logger
from llmling_textual.widgets.pool_list import PoolEntry, PoolList, PoolPreview


if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Key


logger = get_logger(__name__)


class PoolSelectionScreen(ModalScreen[None]):
    """Modal for selecting active pool."""

    DEFAULT_CSS = """
    PoolSelectionScreen {
        align: center middle;
    }

    #pool-container {
        width: 90%;
        height: 90%;
        background: $surface;
        border: thick $background;
    }
    """

    def compose(self) -> ComposeResult:
        """Create screen layout."""
        with Horizontal(id="pool-container"):
            yield PoolList()
            yield PoolPreview()

    def on_pool_entry_selected(self, message: PoolEntry.Selected):
        """Handle pool selection."""
        # Update store
        store = ConfigStore("agents.json")
        store.set_active(message.config.name)

        # Update UI
        for entry in self.query(PoolEntry):
            entry.active = entry.config.name == message.config.name

        # Update preview
        if preview := self.query_one(PoolPreview):
            self.run_worker(preview.show_config(message.config.path))

    async def _on_key(self, event: Key):
        """Handle keyboard input."""
        if event.key == "escape":
            self.app.pop_screen()


if __name__ == "__main__":
    from textualicious import show

    show(PoolSelectionScreen())
