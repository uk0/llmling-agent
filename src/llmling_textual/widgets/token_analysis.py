"""Token widget. Credits to Elia (https://github.com/darrenburns/elia)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import TYPE_CHECKING

from rich.style import Style
from rich.text import Text
from textual.widget import Widget


if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
    from tiktoken import Encoding


@dataclass
class TokenAnalysisRenderable:
    """Renderable for token analysis."""

    tokens: list[int]
    encoder: Encoding

    def __post_init__(self):
        self.parts = self.encoder.decode_tokens_bytes(self.tokens)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        colours = cycle([Style.parse("red"), Style.parse("green"), Style.parse("blue")])
        parts = [Text(part.decode("utf-8"), next(colours)) for part in self.parts]
        text = Text("").join(parts)
        lines = text.wrap(console, width=options.max_width)
        yield lines


class TokenAnalysis(Widget):
    """Widget for token analysis."""

    def __init__(
        self,
        tokens: list[int],
        encoder: Encoding,
        name: str | None = None,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
        disabled: bool = False,
    ):
        super().__init__(
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        self.tokens = tokens
        self.encoder = encoder

    def render(self) -> RenderableType:
        return TokenAnalysisRenderable(self.tokens, self.encoder)
