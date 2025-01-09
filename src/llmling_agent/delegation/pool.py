"""Agent pool management for collaboration."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, overload

from llmling import BaseRegistry, Config, LLMLingError, RuntimeConfig
from typing_extensions import TypeVar

from llmling_agent.agent import Agent, AnyAgent
from llmling_agent.agent.structured import StructuredAgent
from llmling_agent.delegation.controllers import (
    controlled_conversation,
    interactive_controller,
)
from llmling_agent.delegation.router import CallbackRouter
from llmling_agent.log import get_logger
from llmling_agent.models.context import AgentContext
from llmling_agent.tasks import TaskRegistry


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from types import TracebackType

    from psygnal.containers import EventedDict

    from llmling_agent.common_types import OptionalAwaitable, SessionIdType, StrPath
    from llmling_agent.delegation.agentgroup import AgentGroup
    from llmling_agent.delegation.callbacks import DecisionCallback
    from llmling_agent.delegation.router import (
        Decision,
    )
    from llmling_agent.models.agents import AgentConfig, AgentsManifest, WorkerConfig
    from llmling_agent.models.context import ConfirmationCallback
    from llmling_agent.models.messages import ChatMessage
    from llmling_agent.models.task import AgentTask


logger = get_logger(__name__)


TResult = TypeVar("TResult", default=Any)


@dataclass
class AgentResponse[TResult]:
    """Result from an agent's execution."""

    # TODO: replace with Response

    agent_name: str
    """Name of the agent that produced this result"""

    message: ChatMessage[TResult] | None
    """The actual message with content and metadata"""

    timing: float | None = None
    """Time taken by this agent in seconds"""

    error: str | None = None
    """Error message if agent failed"""

    @property
    def success(self) -> bool:
        """Whether the agent completed successfully."""
        return self.error is None

    @property
    def response(self) -> TResult | None:
        """Convenient access to message content."""
        return self.message.content if self.message else None


class AgentPool(BaseRegistry[str, AnyAgent[Any, Any]]):
    """Pool of initialized agents.

    Each agent maintains its own runtime environment based on its configuration.
    """

    def __init__(
        self,
        manifest: AgentsManifest,
        *,
        agents_to_load: list[str] | None = None,
        connect_agents: bool = True,
        confirmation_callback: ConfirmationCallback | None = None,
    ):
        """Initialize agent pool with immediate agent creation.

        Args:
            manifest: Agent configuration manifest
            agents_to_load: Optional list of agent names to initialize
                          If None, all agents from manifest are loaded
            connect_agents: Whether to set up forwarding connections
            confirmation_callback: Handler callback for tool / step confirmations.
        """
        super().__init__()
        from llmling_agent.models.context import AgentContext

        self.manifest = manifest
        self._confirmation_callback = confirmation_callback
        self.exit_stack = AsyncExitStack()

        # Validate requested agents exist
        to_load = set(agents_to_load) if agents_to_load else set(manifest.agents)
        if invalid := (to_load - set(manifest.agents)):
            msg = f"Unknown agents: {', '.join(invalid)}"
            raise ValueError(msg)
        # register tasks
        self._tasks = TaskRegistry()
        # Register tasks from manifest
        for name, task in manifest.tasks.items():
            self._tasks.register(name, task)
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
                confirmation_callback=confirmation_callback,
            )

            # Create agent with runtime and context
            agent = Agent[Any](
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

    async def __aenter__(self) -> Self:
        """Enter async context and initialize all agents."""
        try:
            # Enter async context for all agents
            for agent in self.agents.values():
                await self.exit_stack.enter_async_context(agent)
        except Exception as e:
            await self.cleanup()
            msg = "Failed to initialize agent pool"
            logger.exception(msg, exc_info=e)
            raise RuntimeError(msg) from e
        else:
            return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context."""
        await self.cleanup()

    async def cleanup(self):
        """Clean up all agents."""
        for agent in self.values():
            if agent.runtime:
                await agent.runtime.shutdown()
        await self.exit_stack.aclose()
        self.clear()

    def create_group[TDeps](
        self,
        agents: Sequence[str | AnyAgent[TDeps, Any]],
        *,
        shared_prompt: str | None = None,
        shared_deps: TDeps | None = None,
    ) -> AgentGroup[TDeps]:
        """Create a group from agent names or instances.

        Args:
            agents: List of agent names or instances
            shared_prompt: Optional prompt for all agents
            shared_deps: Optional shared dependencies

        Returns:
            Configured agent group

        Example:
            ```python
            # Mix of names and instances
            group = pool.create_group([
                "agent1",
                existing_agent,
                "agent3"
            ])
            ```
        """
        from llmling_agent.delegation.agentgroup import AgentGroup

        resolved_agents: list[AnyAgent[TDeps, Any]] = [
            agent if isinstance(agent, Agent | StructuredAgent) else self.get_agent(agent)
            for agent in agents
        ]
        return AgentGroup(
            resolved_agents,
            shared_prompt=shared_prompt,
            shared_deps=shared_deps,
        )

    def start_supervision(self) -> OptionalAwaitable[None]:
        """Start supervision interface.

        Can be called either synchronously or asynchronously:

        # Sync usage:
        start_supervision(pool)

        # Async usage:
        await start_supervision(pool)
        """
        from llmling_agent.delegation.supervisor_ui import SupervisorApp

        app = SupervisorApp(self)
        if asyncio.get_event_loop().is_running():
            # We're in an async context
            return app.run_async()
        # We're in a sync context
        app.run()
        return None

    @property
    def agents(self) -> EventedDict[str, AnyAgent[Any, Any]]:
        """Get agents dict (backward compatibility)."""
        return self._items

    @property
    def _error_class(self) -> type[LLMLingError]:
        """Error class for agent operations."""
        return LLMLingError

    def _validate_item(self, item: Agent[Any] | Any) -> Agent[Any]:
        """Validate and convert items before registration.

        Args:
            item: Item to validate

        Returns:
            Validated Agent

        Raises:
            LLMlingError: If item is not a valid agent
        """
        if not isinstance(item, Agent):
            msg = f"Item must be Agent, got {type(item)}"
            raise self._error_class(msg)
        return item

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

    async def create_agent(
        self,
        name: str,
        config: AgentConfig,
        *,
        temporary: bool = True,
    ) -> Agent[Any]:
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
            agent = Agent[Any](
                agent_type=config.type,
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
        agent: Agent[TDeps] | str,
        new_name: str | None = None,
        *,
        model_override: str | None = None,
        system_prompts: list[str] | None = None,
        template_context: dict[str, Any] | None = None,
    ) -> Agent[TDeps]:
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
            original_agent: Agent[TDeps] = self.get_agent(agent)
        else:
            config = agent.context.config  # type: ignore
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
        new_agent = Agent[TDeps](
            runtime=original_agent.runtime,
            context=original_agent.context,
            # result_type=original_agent.actual_type,
            model=new_config.model,  # type: ignore
            system_prompt=new_config.system_prompts,
            name=new_name or f"{config.name}_copy_{len(self.agents)}",
        )

        # Register in pool
        agent_name = new_agent.name
        self.manifest.agents[agent_name] = new_config
        self.agents[agent_name] = new_agent

        return new_agent

    def setup_agent_workers(self, agent: AnyAgent[Any, Any], workers: list[WorkerConfig]):
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

    @overload
    def get_agent[TDeps, TResult](
        self,
        agent: str | Agent[Any],
        *,
        deps: TDeps,
        return_type: type[TResult],
        model_override: str | None = None,
        session_id: SessionIdType = None,
        environment_override: StrPath | Config | None = None,
    ) -> StructuredAgent[TDeps, TResult]: ...

    @overload
    def get_agent[TDeps](
        self,
        agent: str | Agent[Any],
        *,
        deps: TDeps,
        model_override: str | None = None,
        session_id: SessionIdType = None,
        environment_override: StrPath | Config | None = None,
    ) -> Agent[TDeps]: ...

    @overload
    def get_agent[TResult](
        self,
        agent: str | Agent[Any],
        *,
        return_type: type[TResult],
        model_override: str | None = None,
        session_id: SessionIdType = None,
        environment_override: StrPath | Config | None = None,
    ) -> StructuredAgent[Any, TResult]: ...

    @overload
    def get_agent(
        self,
        agent: str | Agent[Any],
        *,
        model_override: str | None = None,
        session_id: SessionIdType = None,
        environment_override: StrPath | Config | None = None,
    ) -> Agent[Any]: ...

    def get_agent[TDeps, TResult](
        self,
        agent: str | Agent[Any],
        *,
        deps: TDeps | None = None,
        return_type: type[TResult] | None = None,
        model_override: str | None = None,
        session_id: SessionIdType = None,
        environment_override: StrPath | Config | None = None,
    ) -> AnyAgent[TDeps, TResult]:
        """Get or wrap an agent.

        Args:
            agent: Either agent name or instance
            deps: Dependencies for the agent
            return_type: Optional type to make agent structured
            model_override: Optional model override
            session_id: Optional session ID to recover conversation
            environment_override: Optional environment configuration:
                - Path to environment file
                - Complete Config instance
                - None to use agent's default environment

        Returns:
            Either regular Agent or StructuredAgent depending on return_type

        Raises:
            KeyError: If agent name not found
            ValueError: If environment configuration is invalid
        """
        # Get base agent
        base = agent if isinstance(agent, Agent) else self.agents[agent]
        if deps is not None:
            base.context = base.context or AgentContext[TDeps].create_default(base.name)
            base.context.data = deps

        # Apply overrides
        if model_override:
            base.set_model(model_override)  # type: ignore

        if session_id:
            base.conversation.load_history_from_database(session_id=session_id)

        if environment_override:
            if isinstance(environment_override, Config):
                base.context.runtime = RuntimeConfig.from_config(environment_override)
            else:
                cfg = Config.from_file(environment_override)
                base.context.runtime = RuntimeConfig.from_config(cfg)

        # Wrap in StructuredAgent if return_type provided
        if return_type is not None:
            return StructuredAgent[Any, TResult](base, return_type)

        return base

    @classmethod
    @asynccontextmanager
    async def open[TDeps, TResult](
        cls,
        config_path: StrPath | AgentsManifest[TDeps, TResult] | None = None,
        *,
        agents: list[str] | None = None,
        connect_agents: bool = True,
        confirmation_callback: ConfirmationCallback | None = None,
    ) -> AsyncIterator[AgentPool]:
        """Open an agent pool from configuration.

        Args:
            config_path: Path to agent configuration file or manifest
            agents: Optional list of agent names to initialize
            connect_agents: Whether to set up forwarding connections
            confirmation_callback: Callback to confirm agent tool selection

        Yields:
            Configured agent pool
        """
        from llmling_agent.models import AgentsManifest

        match config_path:
            case None:
                manifest = AgentsManifest[Any, Any]()
            case str():
                manifest = AgentsManifest[Any, Any].from_file(config_path)
            case AgentsManifest():
                manifest = config_path
            case _:
                msg = f"Invalid config path: {config_path}"
                raise ValueError(msg)
        pool = cls(
            manifest,
            agents_to_load=agents,
            connect_agents=connect_agents,
            confirmation_callback=confirmation_callback,
        )
        try:
            async with pool:
                yield pool
        finally:
            await pool.cleanup()

    async def team_task(
        self,
        prompt: str,
        team: Sequence[str | Agent[Any]],
        *,
        mode: Literal["parallel", "sequential"] = "parallel",
        result_type: type[Any] | None = None,
        model_override: str | None = None,
        environment_override: StrPath | Config | None = None,
    ) -> list[AgentResponse]:
        """Execute a task with a team of agents.

        Args:
            prompt: Task to execute
            team: List of agents or agent names
            mode: Whether to run agents in parallel or sequence
            result_type: Optional type for structured responses
            model_override: Optional model override for all agents
            environment_override: Optional environment override for all agents
        """

        async def run_agent(agent_ref: str | Agent[Any]) -> AgentResponse:
            try:
                agent = (
                    agent_ref
                    if isinstance(agent_ref, Agent)
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
                    agent.runtime = RuntimeConfig.from_config(cfg)
                result = await agent.run(prompt, result_type=result_type)
                return AgentResponse(agent_name=agent.name, message=result)
            except Exception as e:
                name = agent_ref if isinstance(agent_ref, str) else agent_ref.name
                logger.exception("Agent %s failed", name)
                return AgentResponse(agent_name=name, message=None, error=str(e))

        if mode == "parallel":
            tasks = [run_agent(ref) for ref in team]
            return list(await asyncio.gather(*tasks))

        # Sequential execution
        return [await run_agent(ref) for ref in team]

    def list_agents(self) -> list[str]:
        """List available agent names."""
        return list(self.manifest.agents)

    def get_task(self, name: str) -> AgentTask[Any, Any]:
        return self._tasks[name]

    def register_task(self, name: str, task: AgentTask[Any, Any]):
        self._tasks.register(name, task)

    async def controlled_conversation(
        self,
        initial_agent: str | Agent[Any] = "starter",
        initial_prompt: str = "Hello!",
        decision_callback: DecisionCallback = interactive_controller,
    ) -> None:
        """Start a controlled conversation between agents.

        Args:
            initial_agent: Agent instance or name to start with
            initial_prompt: First message to start conversation
            decision_callback: Callback for routing decisions
        """
        await controlled_conversation(
            self,
            initial_agent=initial_agent,
            initial_prompt=initial_prompt,
            decision_callback=decision_callback,
        )

    @overload
    async def controlled_talk(
        self,
        agent: str | Agent[Any],
        message: str,
        decision_callback: DecisionCallback[str] = interactive_controller,
    ) -> tuple[ChatMessage[str], Decision]: ...

    @overload
    async def controlled_talk[TMessage](
        self,
        agent: StructuredAgent[Any, TMessage],
        message: TMessage,
        decision_callback: DecisionCallback[TMessage],
    ) -> tuple[ChatMessage[TMessage], Decision]: ...

    async def controlled_talk[TMessage](
        self,
        agent: str | Agent[Any] | StructuredAgent[Any, TMessage],
        message: str | TMessage,
        decision_callback: DecisionCallback[Any] = interactive_controller,
    ) -> tuple[ChatMessage[Any], Decision]:
        """Get one response with control decision.

        Args:
            agent: Either:
                - Name of agent to look up
                - Regular Agent instance
                - StructuredAgent for type-safe messages
            message: Message to send (type depends on agent)
            decision_callback: Callback for routing decision

        Returns:
            Tuple of (response message, routing decision)
        """
        # Create appropriate controller based on message type
        controller = CallbackRouter[TMessage](self, decision_callback)

        # Get or use agent
        match agent:
            case str():
                # String name - get regular agent
                current_agent: Agent[Any] | StructuredAgent[Any, TMessage] = (
                    self.get_agent(agent)
                )
            case Agent() | StructuredAgent():
                current_agent = agent
            case _:
                msg = f"Invalid agent type: {type(agent)}"
                raise TypeError(msg)

        # Run with message
        response = await current_agent.run(message)  # type: ignore
        decision = await controller.decide(response.content)  # type: ignore

        return response, decision


if __name__ == "__main__":

    async def main():
        path = "src/llmling_agent/config/resources/agents.yml"
        async with AgentPool.open(path) as pool:
            agent: Agent[Any] = pool.get_agent("overseer")
            print(agent)

    import asyncio

    asyncio.run(main())
