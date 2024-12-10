"""Models for agent runners."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentRunConfig:
    """Configuration for running agents."""

    agent_names: list[str]
    """Names of agents to run"""

    prompts: list[str]
    """List of prompts to send to the agent(s)"""

    environment: str | None = None
    """Optional environment override path"""

    model: str | None = None
    """Optional model override"""

    output_format: str = "text"
    """Output format for results (text/json/yaml)"""
