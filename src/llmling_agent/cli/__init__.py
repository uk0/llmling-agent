"""CLI commands for llmling-agent."""

from __future__ import annotations
from llmling.config.store import ConfigStore


# Create stores for environments and agents
config_store = ConfigStore()  # Default for environments
agent_store = ConfigStore("agents.json")  # For agent configurations
