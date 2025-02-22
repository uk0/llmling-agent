"""Database tools."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import sqlalchemy as sa
    from sqlmodel import SQLModel


def generate_db_description(models: list[type[SQLModel]]) -> str:
    """Generate complete database documentation."""
    parts = [
        "Database Schema:",
        "================",
        "",
        "Available tables and their structure:",
        "",
    ]

    from toprompt.sqlmodel_types import generate_schema_description

    # Add each model's documentation
    for model in models:
        parts.append(generate_schema_description(model))
        parts.append("")  # Empty line between tables

    return "\n".join(parts)


class DatabaseQueryTool:
    """Tool for executing SQL queries."""

    def __init__(self, engine: sa.Engine, models: list[type[SQLModel]]):
        self.engine = engine
        self.schema_doc = generate_db_description(models)

    async def query(self, sql: str) -> str:
        """Execute a SQL query and return results.

        The query must be a SELECT statement for safety.
        Returns results formatted as a table.
        """
        import sqlalchemy as sa

        # Basic safety check
        if not sql.lstrip().lower().startswith("select"):
            return "Error: Only SELECT queries are allowed"

        try:
            with self.engine.connect() as conn:
                result = conn.execute(sa.text(sql))

                # Format results as table
                if not result.returns_rows:
                    return "Query executed successfully (no results)"

                # Convert keys to list of strings
                headers = list(result.keys())

                # Create rows list starting with headers
                rows: list[tuple[str, ...]] = [tuple(headers)]

                # Add data rows
                rows.extend(tuple(str(v) for v in row) for row in result)

                # Calculate column widths
                widths = [max(len(str(r[i])) for r in rows) for i in range(len(headers))]

                # Format table
                lines = []
                # Header
                lines.append(
                    "| "
                    + " | ".join(
                        f"{h:<{w}}" for h, w in zip(headers, widths, strict=True)
                    )
                    + " |"
                )
                # Separator
                lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
                # Data rows
                lines.extend([
                    "| "
                    + " | ".join(f"{v!s:<{w}}" for v, w in zip(row, widths, strict=True))
                    + " |"
                    for row in rows[1:]
                ])

                return "\n".join(lines)

        except Exception as e:  # noqa: BLE001
            return f"Error executing query: {e}"

    def get_schema_doc(self) -> str:
        """Get the database schema documentation."""
        return self.schema_doc
