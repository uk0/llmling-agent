"""Database configuration and initialization for LLMling agent."""

from llmling_agent.storage.engine import engine, init_database


# Initialize database on import
init_database()

__all__ = [
    "engine",
    "init_database",
]
