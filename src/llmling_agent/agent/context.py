"""Runtime context models for Agents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import cached_property
import json
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import TypeVar

from llmling_agent.messaging.messagenode import NodeContext
from llmling_agent.prompts.conversion_manager import ConversionManager
from llmling_agent.tools.base import ToolInfo


if TYPE_CHECKING:
    from llmling import RuntimeConfig

    from llmling_agent.agent import AnyAgent
    from llmling_agent.config.capabilities import Capabilities
    from llmling_agent.delegation.pool import AgentPool
    from llmling_agent.models.agents import AgentConfig
    from llmling_agent.prompts.manager import PromptManager
    from llmling_agent.storage import StorageManager


TDeps = TypeVar("TDeps", default=Any)


@dataclass(kw_only=True)
class AgentContext[TDeps](NodeContext[TDeps]):
    """Runtime context for agent execution.

    Generically typed with AgentContext[Type of Dependencies]
    """

    capabilities: Capabilities
    """Current agent's capabilities."""

    config: AgentConfig
    """Current agent's specific configuration."""

    model_settings: dict[str, Any] = field(default_factory=dict)
    """Model-specific settings."""

    data: TDeps | None = None
    """Custom context data."""

    runtime: RuntimeConfig | None = None
    """Reference to the runtime configuration."""

    confirmation_callback: ConfirmationCallback | None = None
    """Optional confirmation handler for tool execution."""

    @classmethod
    def create_default(
        cls,
        name: str,
        capabilities: Capabilities | None = None,
        deps: TDeps | None = None,
        pool: AgentPool | None = None,
    ) -> AgentContext[TDeps]:
        """Create a default agent context with minimal privileges.

        Args:
            name: Name of the agent
            capabilities: Optional custom capabilities (defaults to minimal access)
            deps: Optional dependencies for the agent
            pool:(optional): Optional pool the agent is part of
        """
        from llmling_agent.config.capabilities import Capabilities
        from llmling_agent.models import AgentConfig, AgentsManifest

        caps = capabilities or Capabilities()
        defn = AgentsManifest()
        cfg = AgentConfig(name=name)
        return cls(
            node_name=name,
            capabilities=caps,
            definition=defn,
            config=cfg,
            data=deps,
            pool=pool,
        )

    @cached_property
    def converter(self) -> ConversionManager:
        """Get conversion manager from global config."""
        return ConversionManager(self.definition.conversion)

    # TODO: perhaps add agent directly to context?
    @property
    def agent(self) -> AnyAgent[TDeps, Any]:
        """Get the agent instance from the pool."""
        assert self.pool, "No agent pool available"
        assert self.node_name, "No agent name available"
        return self.pool.agents[self.node_name]

    @cached_property
    def storage(self) -> StorageManager:
        """Storage manager from pool or config."""
        from llmling_agent.storage import StorageManager

        if self.pool:
            return self.pool.storage
        return StorageManager(self.definition.storage)

    @property
    def prompt_manager(self) -> PromptManager:
        """Get prompt manager from manifest."""
        return self.definition.prompt_manager

    async def handle_confirmation(
        self,
        ctx: AgentContext,
        tool: ToolInfo,
        args: dict[str, Any],
    ) -> ConfirmationResult:
        """Handle tool execution confirmation.

        Returns True if:
        - No confirmation handler is set
        - Handler confirms the execution
        """
        if not self.confirmation_callback:
            return "allow"
        result = self.confirmation_callback(ctx, tool, args)
        if isinstance(result, str):
            return result  # pyright: ignore
        return await result


ConfirmationResult = Literal["allow", "skip", "abort_run", "abort_chain"]

ConfirmationCallback = Callable[
    [AgentContext, ToolInfo, dict[str, Any]],
    ConfirmationResult | Awaitable[ConfirmationResult],
]


async def simple_confirmation(
    ctx: AgentContext,
    tool: ToolInfo,
    args: dict[str, Any],
) -> ConfirmationResult:
    """Simple confirmation handler using input() in executor."""
    from pydantic_ai import RunContext

    # Get agent name regardless of context type
    agent_name = ctx.node_name

    prompt = dedent(f"""
        Tool Execution Confirmation
        -------------------------
        Tool: {tool.name}
        Description: {tool.description or "No description"}
        Agent: {agent_name}

        Arguments:
        {json.dumps(args, indent=2)}

        Additional Context:
        """).strip()

    # Add run-specific info if available
    if isinstance(ctx, RunContext):
        prompt += dedent(f"""
            - Model: {ctx.model.name()}
            - Prompt: "{ctx.prompt[:100]}..."
        """)

    prompt += "\nAllow this tool execution? [y/N]: "

    response = input(prompt + "\n")
    # # Run input() in executor to avoid blocking
    # loop = asyncio.get_running_loop()
    # response = await loop.run_in_executor(None, input, prompt + "\n")
    return "allow" if response.lower().startswith("y") else "skip"
