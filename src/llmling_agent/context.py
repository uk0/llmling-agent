from __future__ import annotations

from typing import Any

from llmling import RuntimeConfig  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field

from llmling_agent.config.capabilities import Capabilities
from llmling_agent.models import AgentConfig, AgentsManifest


class AgentContext(BaseModel):
    """Runtime context for agent execution."""

    agent_name: str
    """Name of the current agent."""

    capabilities: Capabilities
    """Current agent's capabilities."""

    definition: AgentsManifest
    """Complete agent definition with all configurations."""

    config: AgentConfig
    """Current agent's specific configuration."""

    model_settings: dict[str, Any] = Field(default_factory=dict)
    """Model-specific settings."""

    runtime: RuntimeConfig | None = None
    """Reference to the runtime configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    def get_capabilities(self) -> Capabilities:
        """Get the current agent's capabilities."""
        return self.capabilities

    @classmethod
    def create_default(
        cls,
        name: str,
        capabilities: Capabilities | None = None,
    ) -> AgentContext:
        """Create a default agent context with minimal privileges.

        Args:
            name: Name of the agent
            capabilities: Optional custom capabilities (defaults to minimal access)
        """
        caps = capabilities or Capabilities(history_access="none", stats_access="none")
        defn = AgentsManifest(responses={}, agents={}, roles={})
        cfg = AgentConfig(name=name, role="assistant")
        return cls(agent_name=name, capabilities=caps, definition=defn, config=cfg)
