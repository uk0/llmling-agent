from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.table import Table
from textual.widgets import Static


if TYPE_CHECKING:
    from typing import Any

    from pydantic import BaseModel
    from rich.style import StyleType


class ModelViewer(Static):
    """Display a Pydantic model instance in a table format."""

    DEFAULT_CSS = """
    ModelViewer {
        padding: 1;
    }
    """

    def __init__(
        self,
        model: BaseModel,
        *,
        show_types: bool = True,
        show_descriptions: bool = True,
        field_style: StyleType | None = None,
        value_style: StyleType | None = None,
        type_style: StyleType = "dim",
        description_style: StyleType = "italic blue",
    ):
        """Initialize the model viewer.

        Args:
            model: Pydantic model instance to display
            show_types: Whether to show field types
            show_descriptions: Whether to show field descriptions
            field_style: Style for field names
            value_style: Style for field values
            type_style: Style for type annotations
            description_style: Style for field descriptions
        """
        super().__init__("")
        self.model = model
        self.show_types = show_types
        self.show_descriptions = show_descriptions
        self.field_style = field_style
        self.value_style = value_style
        self.type_style = type_style
        self.description_style = description_style
        self._update_content()

    def _update_content(self):
        """Update the displayed content."""
        table = Table(show_header=False, pad_edge=False, show_edge=True)

        # Add columns
        table.add_column("Field", style=self.field_style)
        table.add_column("Value", style=self.value_style)
        if self.show_types:
            table.add_column("Type", style=self.type_style)
        if self.show_descriptions:
            table.add_column("Description", style=self.description_style)

        # Add rows for each field
        model_fields = self.model.model_fields
        for name, field in model_fields.items():
            value = getattr(self.model, name)
            row: list[Any] = [name, str(value)]

            if self.show_types:
                type_name = field.annotation.__name__ if field.annotation else "N/A"
                row.append(type_name)

            if self.show_descriptions:
                description = field.description or ""
                row.append(description)

            table.add_row(*row)

        self.update(table)
