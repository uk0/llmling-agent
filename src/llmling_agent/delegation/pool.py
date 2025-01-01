"""Agent pool management for collaboration."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal, Self

from llmling import BaseRegistry, Config, LLMLingError, RuntimeConfig
from pydantic import BaseModel
from typing_extensions import TypeVar

from llmling_agent import LLMlingAgent
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from types import TracebackType
    from uuid import UUID

    from psygnal.containers import EventedDict

    from llmling_agent.common_types import StrPath
    from llmling_agent.models.agents import AgentConfig, AgentsManifest, WorkerConfig
    from llmling_agent.models.context import ConfirmationHandler


logger = get_logger(__name__)


TResult = TypeVar("TResult", default=Any)


class AgentResponse[TResult](BaseModel):
    """Result from an agent's execution."""

    agent_name: str
    """Name of the agent that produced this result"""

    response: TResult
    """The actual response, typed according to the agent's result type"""

    timing: float | None = None
    """Time taken by this agent in seconds"""

    error: str | None = None
    """Error message if agent failed"""

    @property
    def success(self) -> bool:
        """Whether the agent completed successfully."""
        return self.error is None


class AgentPool(BaseRegistry[str, LLMlingAgent[Any, Any]]):
    """Pool of initialized agents.

    Each agent maintains its own runtime environment based on its configuration.
    """

    def __init__(
        self,
        manifest: AgentsManifest,
        *,
        agents_to_load: list[str] | None = None,
        connect_agents: bool = True,
        confirmation_handler: ConfirmationHandler | None = None,
    ):
        """Initialize agent pool with immediate agent creation.

        Args:
            manifest: Agent configuration manifest
            agents_to_load: Optional list of agent names to initialize
                          If None, all agents from manifest are loaded
            connect_agents: Whether to set up forwarding connections
            confirmation_handler: Handler callback for tool / step confirmations.
        """
        super().__init__()
        from llmling_agent.models.context import AgentContext

        self.manifest = manifest
        self._confirmation_handler = confirmation_handler

        # Validate requested agents exist
        to_load = set(agents_to_load) if agents_to_load else set(manifest.agents)
        if invalid := (to_load - set(manifest.agents)):
            msg = f"Unknown agents: {', '.join(invalid)}"
            raise ValueError(msg)

        # Create requested agents immediately using sync initialization
        for name in to_load:
            config = manifest.agents[name]
            # Create runtime without async context
            cfg = config.get_config()
            runtime = RuntimeConfig.from_config(cfg)
            runtime._register_default_components()  # Manual initialization

            # Create context with config path and capabilities
            context = AgentContext[Any](
                agent_name=name,
                capabilities=config.capabilities,
                definition=self.manifest,
                config=config,
                pool=self,
            )

            # Create agent with runtime and context
            agent = LLMlingAgent[Any, Any](
                runtime=runtime,
                context=context,
                result_type=None,  # type: ignore[arg-type]
                model=config.model,  # type: ignore[arg-type]
                system_prompt=config.system_prompts,
                name=name,
            )
            self.register(name, agent)

        # Then set up worker relationships
        for name, config in manifest.agents.items():
            if name in self and config.workers:
                self.setup_agent_workers(self[name], config.workers)

        # Set up forwarding connections
        if connect_agents:
            self._connect_signals()

    # async def initialize(self):
    #     """Initialize all agents asynchronously."""
    #     # Create requested agents
    #     for name in self.to_load:
    #         config = self.manifest.agents[name]
    #         await self.create_agent(name, config, temporary=False)

    #     # Set up forwarding connections
    #     if self._connect_signals:
    #         self._setup_connections()

    @property
    def agents(self) -> EventedDict[str, LLMlingAgent[Any, Any]]:
        """Get agents dict (backward compatibility)."""
        return self._items

    @property
    def _error_class(self) -> type[LLMLingError]:
        """Error class for agent operations."""
        return LLMLingError

    def _validate_item(
        self, item: LLMlingAgent[Any, Any] | Any
    ) -> LLMlingAgent[Any, Any]:
        """Validate and convert items before registration.

        Args:
            item: Item to validate

        Returns:
            Validated LLMlingAgent

        Raises:
            LLMlingError: If item is not a valid agent
        """
        if not isinstance(item, LLMlingAgent):
            msg = f"Item must be LLMlingAgent, got {type(item)}"
            raise self._error_class(msg)
        return item

    async def cleanup(self):
        """Clean up all agents."""
        for agent in self.values():
            if agent._runtime:
                await agent._runtime.shutdown()
        self.clear()

    def _setup_connections(self):
        """Set up forwarding connections between agents."""
        from llmling_agent.models.forward_targets import AgentTarget

        for name, config in self.manifest.agents.items():
            if name not in self.agents:
                continue
            agent = self.agents[name]
            for target in config.forward_to:
                if isinstance(target, AgentTarget):
                    if target.name not in self.agents:
                        msg = f"Forward target {target.name} not loaded for {name}"
                        raise ValueError(msg)
                    target_agent = self.agents[target.name]
                    agent.pass_results_to(target_agent)

    def _connect_signals(self):
        """Set up forwarding connections between agents."""
        from llmling_agent.models.forward_targets import AgentTarget

        for name, config in self.manifest.agents.items():
            if name not in self.agents:
                continue
            agent = self.agents[name]
            for target in config.forward_to:
                if isinstance(target, AgentTarget):
                    if target.name not in self.agents:
                        msg = f"Forward target {target.name} not loaded for {name}"
                        raise ValueError(msg)
                    target_agent = self.agents[target.name]
                    agent.pass_results_to(target_agent)

    async def __aenter__(self) -> Self:
        """Enter async context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Exit async context."""
        await self.cleanup()

    async def create_agent(
        self,
        name: str,
        config: AgentConfig,
        *,
        temporary: bool = True,
    ) -> LLMlingAgent[Any, Any]:
        """Create and register a new agent in the pool."""
        from llmling_agent.models.context import AgentContext

        if name in self.agents:
            msg = f"Agent {name} already exists"
            raise ValueError(msg)

        # Create runtime from agent's config
        cfg = config.get_config()
        async with RuntimeConfig.open(cfg) as runtime:
            # Create context with config path and capabilities
            context = AgentContext[Any](
                agent_name=name,
                capabilities=config.capabilities,
                definition=self.manifest,
                config=config,
                pool=self,
            )

            # Create agent with runtime and context
            agent = LLMlingAgent[Any, Any](
                runtime=runtime,
                context=context,
                result_type=None,  # type: ignore[arg-type]
                model=config.model,  # type: ignore[arg-type]
                system_prompt=config.system_prompts,
                name=name,
            )

            # Set up workers if defined
            if config.workers:
                self.setup_agent_workers(agent, config.workers)

            # Register
            self.agents[name] = agent
            if not temporary:
                self.manifest.agents[name] = config

            return agent

    async def clone_agent[TDeps, TResult](
        self,
        agent: LLMlingAgent[TDeps, TResult] | str,
        new_name: str | None = None,
        *,
        model_override: str | None = None,
        system_prompts: list[str] | None = None,
        template_context: dict[str, Any] | None = None,
    ) -> LLMlingAgent[TDeps, TResult]:
        """Create a copy of an agent.

        Args:
            agent: Agent instance or name to clone
            new_name: Optional name for the clone
            model_override: Optional different model
            system_prompts: Optional different prompts
            template_context: Variables for template rendering

        Returns:
            The new agent instance
        """
        # Get original config
        if isinstance(agent, str):
            if agent not in self.manifest.agents:
                msg = f"Agent {agent} not found"
                raise KeyError(msg)
            config = self.manifest.agents[agent]
            original_agent: LLMlingAgent[TDeps, TResult] = self.get_agent(agent)
        else:
            config = agent._context.config  # type: ignore
            original_agent = agent

        # Create new config
        new_config = config.model_copy(deep=True)

        # Apply overrides
        if model_override:
            new_config.model = model_override
        if system_prompts:
            new_config.system_prompts = system_prompts

        # Handle template rendering
        if template_context:
            new_config.system_prompts = new_config.render_system_prompts(template_context)

        # Create new agent with same runtime
        new_agent = LLMlingAgent[TDeps, TResult](
            runtime=original_agent._runtime,
            context=original_agent._context,
            result_type=original_agent.actual_type,
            model=new_config.model,  # type: ignore
            system_prompt=new_config.system_prompts,
            name=new_name or f"{config.name}_copy_{len(self.agents)}",
        )

        # Register in pool
        agent_name = new_agent.name
        self.manifest.agents[agent_name] = new_config
        self.agents[agent_name] = new_agent

        return new_agent

    def setup_agent_workers(
        self, agent: LLMlingAgent[Any, Any], workers: list[WorkerConfig]
    ):
        """Set up workers for an agent from configuration."""
        for worker_config in workers:
            try:
                worker = self.get_agent(worker_config.name)
                agent.register_worker(
                    worker,
                    name=worker_config.name,
                    reset_history_on_run=worker_config.reset_history_on_run,
                    pass_message_history=worker_config.pass_message_history,
                    share_context=worker_config.share_context,
                )
            except KeyError as e:
                msg = f"Worker agent {worker_config.name!r} not found"
                raise ValueError(msg) from e

    def get_agent(
        self,
        name: str,
        *,
        model_override: str | None = None,
        session_id: str | UUID | None = None,
        environment_override: StrPath | Config | None = None,
    ) -> LLMlingAgent:
        """Get an agent by name with optional runtime modifications.

        Args:
            name: Name of agent to get
            model_override: Optional model to use
            session_id: Optional session id to recover conversation
            environment_override: Optional environment to use

        Returns:
            Requested agent instance

        Raises:
            KeyError: If agent name not found
        """
        if name not in self.agents:
            msg = f"Agent {name} not found"
            raise KeyError(msg)

        agent = self.agents[name]
        if agent._context and agent._context.pool is not self:
            agent._context.pool = self
        if session_id:
            agent.conversation.load_history_from_database(session_id=session_id)
        # Apply any overrides to the existing agent
        if model_override:
            agent.set_model(model_override)  # type: ignore

        if env := environment_override:
            # Create new runtime with override
            cfg = env if isinstance(env, Config) else Config.from_file(env)
            agent._runtime = RuntimeConfig.from_config(cfg)

        return agent

    @classmethod
    @asynccontextmanager
    async def open(
        cls,
        config_path: StrPath | AgentsManifest,
        *,
        agents: list[str] | None = None,
        connect_agents: bool = True,
    ) -> AsyncIterator[AgentPool]:
        """Open an agent pool from configuration.

        Args:
            config_path: Path to agent configuration file or manifest
            agents: Optional list of agent names to initialize
            connect_agents: Whether to set up forwarding connections

        Yields:
            Configured agent pool
        """
        from llmling_agent.models import AgentsManifest

        manifest = (
            config_path
            if isinstance(config_path, AgentsManifest)
            else AgentsManifest.from_file(config_path)
        )
        pool = cls(manifest, agents_to_load=agents, connect_agents=connect_agents)
        try:
            yield pool
        finally:
            await pool.cleanup()

    async def team_task(
        self,
        prompt: str,
        team: Sequence[str | LLMlingAgent[Any, Any]],
        *,
        mode: Literal["parallel", "sequential"] = "parallel",
        model_override: str | None = None,
        environment_override: StrPath | Config | None = None,
    ) -> list[AgentResponse]:
        """Execute a task with a team of agents.

        Args:
            prompt: Task to execute
            team: List of agents or agent names
            mode: Whether to run agents in parallel or sequence
            model_override: Optional model override for all agents
            environment_override: Optional environment override for all agents
        """

        async def run_agent(agent_ref: str | LLMlingAgent[Any, Any]) -> AgentResponse:
            try:
                # Use agent directly if instance provided, otherwise look up by name
                agent = (
                    agent_ref
                    if isinstance(agent_ref, LLMlingAgent)
                    else self.get_agent(agent_ref)
                )
                if model_override:
                    agent.set_model(model_override)  # type: ignore
                if environment_override:
                    cfg = (
                        environment_override
                        if isinstance(environment_override, Config)
                        else Config.from_file(environment_override)
                    )
                    agent._runtime = RuntimeConfig.from_config(cfg)
                result = await agent.run(prompt)
                return AgentResponse(agent_name=agent.name, response=result.data)
            except Exception as e:
                name = agent_ref if isinstance(agent_ref, str) else agent_ref.name
                logger.exception("Agent %s failed", name)
                return AgentResponse(agent_name=name, response=None, error=str(e))

        if mode == "parallel":
            tasks = [run_agent(ref) for ref in team]
            return list(await asyncio.gather(*tasks))

        # Sequential execution
        return [await run_agent(ref) for ref in team]

    def list_agents(self) -> list[str]:
        """List available agent names."""
        return list(self.manifest.agents)


async def main():
    async with AgentPool.open("agents.yml") as pool:
        agent: LLMlingAgent[Any, str] = pool.get_agent("overseer")
        print(agent)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
