from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from llmling_agent.config.capabilities import Capabilities
from llmling_agent.models import AgentConfig, AgentDefinition


class AgentContext(BaseModel):
    """Runtime context for agent execution."""

    agent_name: str
    """Name of the current agent."""

    capabilities: Capabilities
    """Current agent's capabilities."""

    definition: AgentDefinition
    """Complete agent definition with all configurations."""

    config: AgentConfig
    """Current agent's specific configuration."""

    model_settings: dict[str, Any]
    """Model-specific settings."""

    def get_capabilities(self) -> Capabilities:
        """Get the current agent's capabilities."""
        return self.capabilities

    @classmethod
    def create_default(cls, agent_name: str) -> AgentContext:
        """Create a default context with minimal capabilities."""
        return cls(
            agent_name=agent_name,
            capabilities=Capabilities(),  # Default capabilities (all False)
            definition=AgentDefinition(responses={}, agents={}, roles={}),
            config=AgentConfig(name=agent_name, role="assistant"),
            model_settings={},
        )
