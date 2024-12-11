"""Database configuration for LLMling agent."""

from __future__ import annotations

import sqlite3
from typing import Final

from platformdirs import user_data_dir
from sqlmodel import SQLModel, create_engine
from upath import UPath


# App identifiers for platformdirs
APP_NAME: Final = "llmling-agent"
APP_AUTHOR: Final = "llmling"

# Default locations using platformdirs
DEFAULT_DB_NAME: Final = "history.db"
DATA_DIR: Final = UPath(user_data_dir(APP_NAME, APP_AUTHOR))


def get_database_path() -> UPath:
    """Get the database file path, creating directories if needed."""
    db_path = DATA_DIR / DEFAULT_DB_NAME
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


# Create engine with better SQLite settings
engine = create_engine(
    f"sqlite:///{get_database_path()}",
    connect_args={"check_same_thread": False},
    # Enable foreign keys and other SQLite optimizations
    creator=lambda: sqlite3.connect(str(get_database_path()), check_same_thread=False),
)


def init_database() -> None:
    """Initialize database tables."""
    SQLModel.metadata.create_all(engine)


# Initialize tables when module is imported
init_database()
