"""Runtime context models for Agents."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import json
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from llmling import RuntimeConfig  # noqa: TC002
from pydantic_ai import RunContext
from typing_extensions import TypeVar

from llmling_agent.tools.base import ToolInfo


if TYPE_CHECKING:
    from llmling_agent.agent.agent import LLMlingAgent
    from llmling_agent.config.capabilities import Capabilities
    from llmling_agent.delegation.pool import AgentPool
    from llmling_agent.models.agents import AgentConfig, AgentsManifest


TDeps = TypeVar("TDeps", default=Any)


@dataclass
class AgentContext[TDeps]:
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

    model_settings: dict[str, Any] = field(default_factory=dict)
    """Model-specific settings."""

    data: TDeps | None = None
    """Custom context data."""

    runtime: RuntimeConfig | None = None
    """Reference to the runtime configuration."""

    pool: AgentPool | None = None
    """Pool the agent is part of."""

    confirmation_handler: ConfirmationCallback | None = None
    """Optional confirmation handler for tool execution."""

    in_async_context: bool = False
    """Whether we're running in an async context."""

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
        from llmling_agent.config.capabilities import Capabilities
        from llmling_agent.models import AgentConfig, AgentsManifest

        caps = capabilities or Capabilities()
        defn = AgentsManifest[Any, Any](agents={})
        cfg = AgentConfig(name=name)
        return cls(agent_name=name, capabilities=caps, definition=defn, config=cfg)

    # TODO: perhaps add agent directly to context?
    @property
    def agent(self) -> LLMlingAgent[TDeps, Any] | None:
        """Get the agent instance from the pool."""
        if not self.pool or not self.agent_name:
            return None
        return self.pool.get_agent(self.agent_name)

    async def handle_confirmation(
        self,
        run_ctx: RunContext[AgentContext],
        tool: ToolInfo,
        args: dict[str, Any],
    ) -> bool:
        """Handle tool execution confirmation.

        Returns True if:
        - No confirmation handler is set
        - Handler confirms the execution
        """
        if not self.confirmation_handler:
            return True

        result = self.confirmation_handler(run_ctx, tool, args)
        if isinstance(result, bool):
            return result
        return await result


ConfirmationCallback = Callable[
    [RunContext[AgentContext], ToolInfo, dict[str, Any]], Awaitable[bool] | bool
]


async def simple_confirmation(
    ctx: RunContext[AgentContext],
    tool: ToolInfo,
    args: dict[str, Any],
) -> bool:
    """Simple confirmation handler using input() in executor."""
    prompt = dedent(f"""
        Tool Execution Confirmation
        -------------------------
        Tool: {tool.name}
        Description: {tool.description or "No description"}

        Arguments:
        {json.dumps(args, indent=2)}

        Context:
        - Agent: {ctx.deps.agent_name}
        - Model: {ctx.model.name()}
        - Prompt: "{ctx.prompt[:100]}..."

        Allow this tool execution? [y/N]: """).strip()

    # Run input() in executor to avoid blocking
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, input, prompt + "\n")
    return response.lower().startswith("y")
