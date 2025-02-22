"""Logging configuration for llmling_agent."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    from slashed import OutputWriter


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given name.

    Args:
        name: The name of the logger, will be prefixed with 'llmling_agent.'

    Returns:
        A logger instance
    """
    return logging.getLogger(f"llmling_agent.{name}")


@contextmanager
def set_handler_level(
    level: int,
    logger_names: Sequence[str],
    *,
    session_handler: OutputWriter | None = None,
):
    """Temporarily set logging level and optionally add session handler.

    Args:
        level: Logging level to set
        logger_names: Names of loggers to configure
        session_handler: Optional output writer for session logging
    """
    loggers = [logging.getLogger(name) for name in logger_names]
    old_levels = [logger.level for logger in loggers]

    handler = None
    if session_handler:
        from slashed.log import SessionLogHandler

        handler = SessionLogHandler(session_handler)
        for logger in loggers:
            logger.addHandler(handler)

    try:
        for logger in loggers:
            logger.setLevel(level)
        yield
    finally:
        for logger, old_level in zip(loggers, old_levels, strict=True):
            logger.setLevel(old_level)
            if handler:
                logger.removeHandler(handler)
