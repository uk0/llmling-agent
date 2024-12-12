from __future__ import annotations

from datetime import timedelta
from typing import cast

from llmling_agent.history.models import GroupBy


def parse_time_period(period: str) -> timedelta:
    """Parse time period string into timedelta.

    Examples: 1h, 2d, 1w, 1m, 1y

    Raises:
        ValueError: If period format is invalid
    """
    unit = period[-1].lower()
    try:
        value = int(period[:-1])
    except ValueError as e:
        msg = f"Invalid period format: {period}"
        raise ValueError(msg) from e

    match unit:
        case "h":
            return timedelta(hours=value)
        case "d":
            return timedelta(days=value)
        case "w":
            return timedelta(weeks=value)
        case "m":
            return timedelta(days=value * 30)
        case "y":
            return timedelta(days=value * 365)
        case _:
            msg = f"Invalid time unit: {unit}"
            raise ValueError(msg)


def validate_group_by(value: str) -> GroupBy:
    """Validate and convert group by parameter.

    Args:
        value: Group by value to validate

    Returns:
        Validated GroupBy literal

    Raises:
        ValueError: If value is not a valid grouping
    """
    if value not in ("agent", "model", "hour", "day"):
        msg = f"Invalid group_by: {value}"
        raise ValueError(msg)
    return cast(GroupBy, value)
