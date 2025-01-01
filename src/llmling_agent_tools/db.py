from __future__ import annotations

from textwrap import dedent

import sqlalchemy as sa
from sqlmodel import SQLModel


def generate_schema_description(model: type[SQLModel]) -> str:
    """Generate human-readable schema documentation from SQLModel."""
    table = model.__table__  # type: ignore
    fields: list[str] = []

    # Get primary keys
    pk_cols = [c.name for c in table.primary_key.columns]

    # Get field descriptions from model
    docs = {n: f.description for n, f in model.model_fields.items() if f.description}
    # Process each column
    for column in table.columns:
        parts = []
        # Name and type
        parts.append(f"- {column.name}: {column.type}")

        # Add docstring if available
        if doc := docs.get(column.name):
            parts.append(f"\n  Description: {doc}")

        # Add column properties
        properties = []
        if column.name in pk_cols:
            properties.append("primary key")
        if column.foreign_keys:
            fks = [f"references {fk.column.table.name}" for fk in column.foreign_keys]
            properties.append(f"foreign key ({', '.join(fks)})")
        if not column.nullable:
            properties.append("not null")
        if column.default:
            properties.append(f"default: {column.default.arg}")
        if column.server_default:
            properties.append("has server default")
        if properties:
            parts.append(f"  ({', '.join(properties)})")

        fields.append(" ".join(parts))

    # Build complete description
    return dedent(f"""
        Table: {table.name}
        Description: {model.__doc__ or "No description available"}

        Fields:
        {chr(10).join(fields)}
        """)


def generate_db_description(models: list[type[SQLModel]]) -> str:
    """Generate complete database documentation."""
    parts = [
        "Database Schema:",
        "================",
        "",
        "Available tables and their structure:",
        "",
    ]

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
                    "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |"
                )
                # Separator
                lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
                # Data rows
                lines.extend([
                    "| " + " | ".join(f"{v!s:<{w}}" for v, w in zip(row, widths)) + " |"
                    for row in rows[1:]
                ])

                return "\n".join(lines)

        except Exception as e:  # noqa: BLE001
            return f"Error executing query: {e}"

    def get_schema_doc(self) -> str:
        """Get the database schema documentation."""
        return self.schema_doc


if __name__ == "__main__":
    from sqlmodel import create_engine

    from llmling_agent.storage.models import CommandHistory

    # Create tool
    engine = create_engine("sqlite:///blog.db")
    db_tool = DatabaseQueryTool(engine, [CommandHistory])
    print(db_tool.get_schema_doc())
    # Register with agent
    # agent.tools.register_tool(
    #     db_tool.query,
    #     name="query_database",
    #     description=dedent("""
    #         Execute SQL SELECT queries against the database.

    #         Schema Information:
    #         ------------------
    #         {db_tool.get_schema_doc()}
    #     """),
    # )
