"""Runtime context models for Agents."""

from __future__ import annotations

from typing import Any

from llmling import RuntimeConfig  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypeVar

from llmling_agent.config.capabilities import Capabilities
from llmling_agent.models import AgentConfig, AgentsManifest


TDeps = TypeVar("TDeps", default=Any)


class AgentContext[TDeps](BaseModel):
    """Runtime context for agent execution.

    Generically typed with AgentContext[Type of Dependencies]
    """

    agent_name: str
    """Name of the current agent."""

    capabilities: Capabilities
    """Current agent's capabilities."""

    definition: AgentsManifest
    """Complete agent definition with all configurations."""

    config: AgentConfig
    """Current agent's specific configuration."""

    current_prompt: str | None = None
    """Current prompt text for the agent."""

    model_settings: dict[str, Any] = Field(default_factory=dict)
    """Model-specific settings."""

    data: TDeps | None = None
    """Custom context data."""

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
        caps = capabilities or Capabilities()
        defn = AgentsManifest(agents={})
        cfg = AgentConfig(name=name)
        return cls(agent_name=name, capabilities=caps, definition=defn, config=cfg)
