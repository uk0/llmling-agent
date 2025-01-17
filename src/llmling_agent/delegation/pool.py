"""Agent pool management for collaboration."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, Unpack, overload

from llmling import BaseRegistry, LLMLingError, RuntimeConfig
from typing_extensions import TypeVar

from llmling_agent.agent import Agent, AnyAgent
from llmling_agent.agent.connection import TeamTalk
from llmling_agent.agent.structured import StructuredAgent
from llmling_agent.delegation.controllers import interactive_controller
from llmling_agent.log import get_logger
from llmling_agent.models.context import AgentContext
from llmling_agent.tasks import TaskRegistry


if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from psygnal.containers import EventedDict

    from llmling_agent.agent.agent import AgentKwargs
    from llmling_agent.common_types import OptionalAwaitable, SessionIdType, StrPath
    from llmling_agent.delegation.agentgroup import Team
    from llmling_agent.delegation.callbacks import DecisionCallback
    from llmling_agent.models.agents import AgentConfig, AgentsManifest, WorkerConfig
    from llmling_agent.models.context import ConfirmationCallback
    from llmling_agent.models.messages import ChatMessage
    from llmling_agent.models.session import SessionQuery
    from llmling_agent.models.task import AgentTask
    from llmling_agent.responses.models import ResponseDefinition


logger = get_logger(__name__)


TResult = TypeVar("TResult", default=Any)
TPoolDeps = TypeVar("TPoolDeps", default=None)


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


class AgentPool[TPoolDeps](BaseRegistry[str, AnyAgent[Any, Any]]):
    """Pool for managing multiple agents with shared dependencies.

    The pool acts as a central registry and dependency provider for agents.
    By default, all agents share the pool's dependencies, but individual
    agents can override with custom dependencies.

    Generic Parameters:
        TPoolDeps: Type of shared dependencies used across agents.
                   Can be None if no shared dependencies are needed.
    """

    def __init__(
        self,
        manifest: StrPath | AgentsManifest[Any, TResult] | None = None,
        *,
        shared_deps: TPoolDeps | None = None,
        agents_to_load: list[str] | None = None,
        connect_agents: bool = True,
        confirmation_callback: ConfirmationCallback | None = None,
    ):
        """Initialize agent pool with immediate agent creation.

        Args:
            manifest: Agent configuration manifest
            shared_deps: Dependencies to share across all agents
            agents_to_load: Optional list of agent names to initialize
                          If None, all agents from manifest are loaded
            connect_agents: Whether to set up forwarding connections
            confirmation_callback: Handler callback for tool / step confirmations

        Raises:
            ValueError: If manifest contains invalid agent configurations
            RuntimeError: If agent initialization fails
        """
        super().__init__()
        from llmling_agent.models.agents import AgentsManifest
        from llmling_agent.storage import StorageManager

        match manifest:
            case None:
                self.manifest = AgentsManifest[Any, Any]()
            case str():
                self.manifest = AgentsManifest[Any, Any].from_file(manifest)
            case AgentsManifest():
                self.manifest = manifest
            case _:
                msg = f"Invalid config path: {manifest}"
                raise ValueError(msg)
        self.shared_deps = shared_deps
        self._confirmation_callback = confirmation_callback
        self.exit_stack = AsyncExitStack()
        self.storage = StorageManager(self.manifest.storage)

        # Validate requested agents exist
        to_load = set(agents_to_load) if agents_to_load else set(self.manifest.agents)
        if invalid := (to_load - set(self.manifest.agents)):
            msg = f"Unknown agents: {', '.join(invalid)}"
            raise ValueError(msg)
        # register tasks
        self._tasks = TaskRegistry()
        # Register tasks from manifest
        for name, task in self.manifest.tasks.items():
            self._tasks.register(name, task)
        self.pool_talk = TeamTalk.from_agents(list(self.agents.values()))
        # Create requested agents immediately using sync initialization
        for name in to_load:
            agent = self.manifest.get_agent(name, deps=shared_deps)
            self.register(name, agent)

        # Then set up worker relationships
        for name, config in self.manifest.agents.items():
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
    ):
        """Exit async context."""
        await self.cleanup()

    async def cleanup(self):
        """Clean up all agents."""
        for agent in self.values():
            if agent.runtime:
                await agent.runtime.shutdown()
        await self.exit_stack.aclose()
        self.clear()

    @overload
    def create_group[TDeps](
        self,
        agents: Sequence[str | AnyAgent[TDeps, Any]] | None = None,
        *,
        model_override: str | None = None,
        shared_prompt: str | None = None,
    ) -> Team[TDeps]: ...

    @overload
    def create_group(
        self,
        agents: Sequence[str | AnyAgent[Any, Any]] | None = None,
        *,
        model_override: str | None = None,
        shared_prompt: str | None = None,
    ) -> Team[Any]: ...  # Fallback for mixed agent dep types

    def create_group(
        self,
        agents: Sequence[str | AnyAgent[Any, Any]] | None = None,
        *,
        model_override: str | None = None,
        shared_prompt: str | None = None,
    ) -> Team[Any]:
        """Create a group from agent names or instances.

        Args:
            agents: List of agent names or instances (all if None)
            model_override: Optional model to use for all agents
            shared_prompt: Optional prompt for all agents
        """
        from llmling_agent.delegation.agentgroup import Team

        if agents is None:
            agents = list(self.agents.keys())

        # First resolve/configure agents
        resolved_agents: list[AnyAgent[Any, Any]] = []
        for agent in agents:
            if isinstance(agent, str):
                agent = self.get_agent(agent, model_override=model_override)
            resolved_agents.append(agent)

        return Team(agents=resolved_agents, shared_prompt=shared_prompt)

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

    def _validate_item(self, item: AnyAgent[Any, Any] | Any) -> AnyAgent[Any, Any]:
        """Validate and convert items before registration.

        Args:
            item: Item to validate

        Returns:
            Validated Agent

        Raises:
            LLMlingError: If item is not a valid agent
        """
        if not isinstance(item, Agent | StructuredAgent):
            msg = f"Item must be Agent, got {type(item)}"
            raise self._error_class(msg)
        return item

    def _setup_connections(self):
        """Set up forwarding connections between agents."""
        from llmling_agent.models.forward_targets import AgentConnectionConfig

        for name, config in self.manifest.agents.items():
            if name not in self.agents:
                continue
            agent = self.agents[name]
            for target in config.connections:
                if isinstance(target, AgentConnectionConfig):
                    if target.name not in self.agents:
                        msg = f"Forward target {target.name} not loaded for {name}"
                        raise ValueError(msg)
                    target_agent = self.agents[target.name]
                    agent.pass_results_to(target_agent)

    def _connect_signals(self):
        """Set up forwarding connections between agents."""
        from llmling_agent.models.forward_targets import AgentConnectionConfig

        for name, config in self.manifest.agents.items():
            if name not in self.agents:
                continue
            agent = self.agents[name]
            for target in config.connections:
                if isinstance(target, AgentConnectionConfig):
                    if target.name not in self.agents:
                        msg = f"Forward target {target.name} not loaded for {name}"
                        raise ValueError(msg)
                    target_agent = self.agents[target.name]
                    agent.pass_results_to(
                        target_agent,
                        connection_type=target.connection_type,
                    )

    async def create_agent(
        self,
        name: str,
        config: AgentConfig,
        *,
        temporary: bool = True,
    ) -> Agent[Any]:
        """Create and register a new agent in the pool.

        Args:
            name: Name of the new agent
            config: Agent configuration
            temporary: If True, agent won't be added to manifest

        Returns:
            Created and initialized agent

        Raises:
            ValueError: If agent name already exists
            RuntimeError: If agent initialization fails
        """
        from llmling_agent.models.context import AgentContext

        if name in self.agents:
            msg = f"Agent {name} already exists"
            raise ValueError(msg)

        try:
            # Create runtime from agent's config
            cfg = config.get_config()
            runtime = RuntimeConfig.from_config(cfg)

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
                provider=config.get_provider(),
                runtime=runtime,
                context=context,
                result_type=None,  # type: ignore[arg-type]
                model=config.model,  # type: ignore[arg-type]
                system_prompt=config.system_prompts,
                name=name,
            )

            # Enter agent's async context through pool's exit stack
            agent = await self.exit_stack.enter_async_context(agent)

            # Set up workers if defined
            if config.workers:
                self.setup_agent_workers(agent, config.workers)

            # Register in pool and optionally manifest
            self.agents[name] = agent
            if not temporary:
                self.manifest.agents[name] = config
        except Exception as e:
            msg = f"Failed to create agent {name}"
            raise RuntimeError(msg) from e
        else:
            return agent

    @overload
    async def clone_agent[TDeps](
        self,
        agent: str | Agent[TDeps],
        new_name: str | None = None,
        *,
        model_override: str | None = None,
        system_prompts: list[str] | None = None,
        template_context: dict[str, Any] | None = None,
    ) -> Agent[TDeps]: ...

    @overload
    async def clone_agent[TDeps, TResult](
        self,
        agent: StructuredAgent[TDeps, TResult],
        new_name: str | None = None,
        *,
        model_override: str | None = None,
        system_prompts: list[str] | None = None,
        template_context: dict[str, Any] | None = None,
    ) -> StructuredAgent[TDeps, TResult]: ...

    async def clone_agent[TDeps, TAgentResult](
        self,
        agent: str | AnyAgent[TDeps, TAgentResult],
        new_name: str | None = None,
        *,
        model_override: str | None = None,
        system_prompts: list[str] | None = None,
        template_context: dict[str, Any] | None = None,
    ) -> AnyAgent[TDeps, TAgentResult]:
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
            original_agent: AnyAgent[Any, Any] = self.get_agent(agent)
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
        if isinstance(original_agent, StructuredAgent):
            new_agent = new_agent.to_structured(original_agent.actual_type)

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
    def get_agent(
        self,
        agent: str | Agent[Any],
        *,
        model_override: str | None = None,
        session: SessionIdType | SessionQuery = None,
    ) -> Agent[TPoolDeps]: ...

    @overload
    def get_agent[TResult](
        self,
        agent: str | Agent[Any],
        *,
        return_type: type[TResult],
        model_override: str | None = None,
        session: SessionIdType | SessionQuery = None,
    ) -> StructuredAgent[TPoolDeps, TResult]: ...

    @overload
    def get_agent[TCustomDeps](
        self,
        agent: str | Agent[Any],
        *,
        deps: TCustomDeps,
        model_override: str | None = None,
        session: SessionIdType | SessionQuery = None,
    ) -> Agent[TCustomDeps]: ...

    @overload
    def get_agent[TCustomDeps, TResult](
        self,
        agent: str | Agent[Any],
        *,
        deps: TCustomDeps,
        return_type: type[TResult],
        model_override: str | None = None,
        session: SessionIdType | SessionQuery = None,
    ) -> StructuredAgent[TCustomDeps, TResult]: ...

    def get_agent(
        self,
        agent: str | Agent[Any],
        *,
        deps: Any | None = None,
        return_type: Any | None = None,
        model_override: str | None = None,
        session: SessionIdType | SessionQuery = None,
    ) -> AnyAgent[Any, Any]:
        """Get or configure an agent from the pool.

        This method provides flexible agent configuration with dependency injection:
        - Without deps: Agent uses pool's shared dependencies
        - With deps: Agent uses provided custom dependencies
        - With return_type: Returns a StructuredAgent with type validation

        Args:
            agent: Either agent name or instance
            deps: Optional custom dependencies (overrides shared deps)
            return_type: Optional type for structured responses
            model_override: Optional model override
            session: Optional session ID or query to recover conversation

        Returns:
            Either:
            - Agent[TPoolDeps] when using pool's shared deps
            - Agent[TCustomDeps] when custom deps provided
            - StructuredAgent when return_type provided

        Raises:
            KeyError: If agent name not found
            ValueError: If configuration is invalid
        """
        # Get base agent
        base = agent if isinstance(agent, Agent) else self.agents[agent]

        # Setup context and dependencies
        if base.context is None:
            base.context = AgentContext[Any].create_default(base.name)

        # Use custom deps if provided, otherwise use shared deps
        base.context.data = deps if deps is not None else self.shared_deps

        # Apply overrides
        if model_override:
            base.set_model(model_override)

        if session:
            base.conversation.load_history_from_database(session=session)

        # Convert to structured if needed
        if return_type is not None:
            return base.to_structured(return_type)

        return base

    def list_agents(self) -> list[str]:
        """List available agent names."""
        return list(self.manifest.agents)

    def get_task(self, name: str) -> AgentTask[Any, Any]:
        return self._tasks[name]

    def register_task(self, name: str, task: AgentTask[Any, Any]):
        self._tasks.register(name, task)

    @overload
    async def add_agent(
        self,
        name: str,
        *,
        result_type: None = None,
        **kwargs: Unpack[AgentKwargs],
    ) -> Agent[Any]: ...

    @overload
    async def add_agent[TResult](
        self,
        name: str,
        *,
        result_type: type[TResult] | str | ResponseDefinition,
        **kwargs: Unpack[AgentKwargs],
    ) -> StructuredAgent[Any, TResult]: ...

    async def add_agent(
        self,
        name: str,
        *,
        result_type: type[Any] | str | ResponseDefinition | None = None,
        **kwargs: Unpack[AgentKwargs],
    ) -> Agent[Any] | StructuredAgent[Any, Any]:
        """Add a new permanent agent to the pool.

        Args:
            name: Name for the new agent
            result_type: Optional type for structured responses:
                - None: Regular unstructured agent
                - type: Python type for validation
                - str: Name of response definition
                - ResponseDefinition: Complete response definition
            **kwargs: Additional agent configuration

        Returns:
            Either a regular Agent or StructuredAgent depending on result_type
        """
        agent = Agent(name=name, **kwargs)
        agent = await self.exit_stack.enter_async_context(agent)

        # Register in pool
        self.agents[name] = agent

        # Convert to structured if needed
        if result_type is not None:
            return agent.to_structured(result_type)
        return agent

    async def controlled_conversation(
        self,
        initial_agent: str | Agent[Any] = "starter",
        initial_prompt: str = "Hello!",
        decision_callback: DecisionCallback = interactive_controller,
    ):
        """Start a controlled conversation between agents.

        Args:
            initial_agent: Agent instance or name to start with
            initial_prompt: First message to start conversation
            decision_callback: Callback for routing decisions
        """
        from llmling_agent.delegation.agentgroup import Team

        group = Team(list(self.agents.values()))

        await group.run_controlled(
            prompt=initial_prompt,
            initial_agent=initial_agent,
            decision_callback=decision_callback,
        )


if __name__ == "__main__":

    async def main():
        path = "src/llmling_agent/config/resources/agents.yml"
        async with AgentPool[None](path) as pool:
            agent: Agent[Any] = pool.get_agent("overseer")
            print(agent)

    import asyncio

    asyncio.run(main())
