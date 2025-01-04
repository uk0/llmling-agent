"""Routing configuration and decision models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel


# Decision models
class Decision(BaseModel):
    """Base class for all control decisions."""

    type: Literal["route", "talk_back", "end"]
    reason: str


class RouteDecision(Decision):
    """Decision to forward to another agent without waiting."""

    type: Literal["route"] = "route"
    target_agent: str


class TalkBackDecision(Decision):
    """Decision to route and wait for response."""

    type: Literal["talk_back"] = "talk_back"
    target_agent: str


class EndDecision(Decision):
    """Decision to end conversation."""

    type: Literal["end"] = "end"


# Routing configuration
@dataclass
class RoutingRule:
    """Single routing rule configuration."""

    keyword: str
    target: str
    reason: str
    wait_for_response: bool = True
    priority: int = 100
    requires_capability: str | None = None


@dataclass
class RoutingConfig:
    """Complete routing configuration."""

    rules: list[RoutingRule]
    default_target: str | None = None
    default_reason: str = "No specific rule matched"
    case_sensitive: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingConfig:
        rules = [RoutingRule(**rule) for rule in data.get("rules", [])]
        return cls(
            rules=rules,
            default_target=data.get("default_target"),
            default_reason=data.get("default_reason", "No specific rule matched"),
            case_sensitive=data.get("case_sensitive", False),
        )
