"""Database engine configuration."""

from __future__ import annotations

import sqlite3
from typing import Final

from platformdirs import user_data_dir
from sqlmodel import create_engine
from upath import UPath


# App identifiers
APP_NAME: Final = "llmling-agent"
APP_AUTHOR: Final = "llmling"

# Default locations
DATA_DIR: Final = UPath(user_data_dir(APP_NAME, APP_AUTHOR))
DEFAULT_DB_NAME: Final = "history.db"


def get_database_path() -> UPath:
    """Get the database file path, creating directories if needed."""
    db_path = DATA_DIR / DEFAULT_DB_NAME
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


# Create engine with better SQLite settings
engine = create_engine(
    f"sqlite:///{get_database_path()}",
    connect_args={"check_same_thread": False},
    creator=lambda: sqlite3.connect(str(get_database_path()), check_same_thread=False),
)


def init_database() -> None:
    """Initialize database tables."""
    # Import here to avoid circular imports
    from llmling_agent.storage.models import SQLModel

    SQLModel.metadata.create_all(engine)
