"""The main Agent. Can do all sort of crazy things."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
import os
from os import PathLike
import time
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict, cast, overload
from uuid import uuid4

from llmling import Config, LLMCallableTool, RuntimeConfig, ToolError
import logfire
from psygnal import Signal
from typing_extensions import TypeVar

from llmling_agent.agent.conversation import ConversationManager
from llmling_agent.log import get_logger
from llmling_agent.messaging.messagenode import MessageNode
from llmling_agent.messaging.messages import ChatMessage, TokenCost
from llmling_agent.models import AgentsManifest
from llmling_agent.models.agents import ToolCallInfo
from llmling_agent.models.session import MemoryConfig, SessionQuery
from llmling_agent.prompts.builtin_provider import RuntimePromptProvider
from llmling_agent.prompts.convert import convert_prompts
from llmling_agent.talk.stats import MessageStats
from llmling_agent.tools.manager import ToolManager
from llmling_agent.utils.inspection import call_with_context, has_return_type
from llmling_agent.utils.result_utils import to_type
from llmling_agent.utils.tasks import TaskManagerMixin


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

    from llmling.config.models import Resource
    from llmling.prompts import PromptType
    import PIL.Image
    from toprompt import AnyPromptType

    from llmling_agent.agent import AgentContext, AnyAgent
    from llmling_agent.agent.context import ConfirmationCallback
    from llmling_agent.agent.interactions import Interactions
    from llmling_agent.agent.structured import StructuredAgent
    from llmling_agent.common_types import (
        AgentName,
        EndStrategy,
        ModelType,
        SessionIdType,
        StrPath,
        ToolType,
    )
    from llmling_agent.config.capabilities import Capabilities
    from llmling_agent.delegation.team import Team
    from llmling_agent.delegation.teamrun import TeamRun
    from llmling_agent.models.mcp_server import MCPServerConfig
    from llmling_agent.models.providers import ProcessorCallback
    from llmling_agent.models.result_types import ResponseDefinition
    from llmling_agent.models.task import Job
    from llmling_agent.tools.base import ToolInfo
    from llmling_agent_providers.base import AgentProvider, StreamingResponseProtocol

    AgentType = (
        Literal["pydantic_ai", "human", "litellm"] | AgentProvider | Callable[..., Any]
    )

logger = get_logger(__name__)

TResult = TypeVar("TResult", default=str)
TDeps = TypeVar("TDeps", default=None)

JINJA_PROC = "jinja_template"  # Name of builtin LLMling Jinja2 processor


class AgentKwargs(TypedDict, total=False):
    # Core Identity
    provider: AgentType
    description: str | None

    # Model Configuration
    model: ModelType
    system_prompt: str | Sequence[str]
    # model_settings: dict[str, Any]

    # Runtime Environment
    runtime: RuntimeConfig | Config | StrPath | None
    tools: Sequence[ToolType] | None
    capabilities: Capabilities | None
    mcp_servers: Sequence[str | MCPServerConfig] | None

    # Execution Settings
    retries: int
    result_retries: int | None
    end_strategy: EndStrategy
    defer_model_check: bool

    # Context & State
    context: AgentContext[Any] | None  # x
    session: SessionIdType | SessionQuery | MemoryConfig | bool | int

    # Behavior Control
    confirmation_callback: ConfirmationCallback | None
    debug: bool


class Agent[TDeps](MessageNode[TDeps, str], TaskManagerMixin):
    """Agent for AI-powered interaction with LLMling resources and tools.

    Generically typed with: LLMLingAgent[Type of Dependencies, Type of Result]

    This agent integrates LLMling's resource system with PydanticAI's agent capabilities.
    It provides:
    - Access to resources through RuntimeConfig
    - Tool registration for resource operations
    - System prompt customization
    - Signals
    - Message history management
    - Database logging
    """

    @dataclass(frozen=True)
    class AgentReset:
        """Emitted when agent is reset."""

        agent_name: AgentName
        previous_tools: dict[str, bool]
        new_tools: dict[str, bool]
        timestamp: datetime = field(default_factory=datetime.now)

    # this fixes weird mypy issue
    conversation: ConversationManager
    talk: Interactions
    tool_used = Signal(ToolCallInfo)
    model_changed = Signal(object)  # Model | None
    chunk_streamed = Signal(str, str)  # (chunk, message_id)
    run_failed = Signal(str, Exception)
    agent_reset = Signal(AgentReset)

    def __init__(
        # we dont use AgentKwargs here so that we can work with explicit ones in the ctor
        self,
        name: str = "llmling-agent",
        provider: AgentType = "pydantic_ai",
        *,
        model: ModelType = None,
        runtime: RuntimeConfig | Config | StrPath | None = None,
        context: AgentContext[TDeps] | None = None,
        session: SessionIdType | SessionQuery | MemoryConfig | bool | int = None,
        system_prompt: AnyPromptType | Sequence[AnyPromptType] = (),
        description: str | None = None,
        tools: Sequence[ToolType] | None = None,
        capabilities: Capabilities | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        resources: Sequence[Resource | PromptType | str] = (),
        retries: int = 1,
        result_retries: int | None = None,
        end_strategy: EndStrategy = "early",
        defer_model_check: bool = False,
        confirmation_callback: ConfirmationCallback | None = None,
        parallel_init: bool = True,
        debug: bool = False,
    ):
        """Initialize agent with runtime configuration.

        Args:
            runtime: Runtime configuration providing access to resources/tools
            context: Agent context with capabilities and configuration
            provider: Agent type to use (ai: PydanticAIProvider, human: HumanProvider)
            session: Memory configuration.
                - None: Default memory config
                - False: Disable message history (max_messages=0)
                - int: Max tokens for memory
                - str/UUID: Session identifier
                - SessionQuery: Query to recover conversation
                - MemoryConfig: Complete memory configuration
            model: The default model to use (defaults to GPT-4)
            system_prompt: Static system prompts to use for this agent
            name: Name of the agent for logging
            description: Description of the Agent ("what it can do")
            tools: List of tools to register with the agent
            capabilities: Capabilities for the agent
            mcp_servers: MCP servers to connect to
            resources: Additional resources to load
            retries: Default number of retries for failed operations
            result_retries: Max retries for result validation (defaults to retries)
            end_strategy: Strategy for handling tool calls that are requested alongside
                          a final result
            defer_model_check: Whether to defer model evaluation until first run
            confirmation_callback: Callback for confirmation prompts
            parallel_init: Whether to initialize resources in parallel
            debug: Whether to enable debug mode
        """
        from llmling_agent.agent import AgentContext, AgentLogger
        from llmling_agent.agent.interactions import Interactions
        from llmling_agent.agent.sys_prompts import SystemPrompts
        from llmling_agent_providers.base import AgentProvider

        self._infinite = False
        # save some stuff for asnyc init
        self._owns_runtime = False
        # prepare context
        ctx = context or AgentContext[TDeps].create_default(name)
        ctx.confirmation_callback = confirmation_callback
        if capabilities is not None:
            ctx.capabilities = capabilities
        super().__init__(name=name, context=ctx, description=description)
        memory_cfg = (
            session
            if isinstance(session, MemoryConfig)
            else MemoryConfig.from_value(session)
        )
        # Initialize runtime
        match runtime:
            case None:
                ctx.runtime = RuntimeConfig.from_config(Config())
            case Config() | str() | PathLike():
                ctx.runtime = RuntimeConfig.from_config(runtime)
            case RuntimeConfig():
                ctx.runtime = runtime

        runtime_provider = RuntimePromptProvider(ctx.runtime)
        ctx.definition.prompt_manager.providers["runtime"] = runtime_provider
        # Initialize tool manager
        all_tools = list(tools or [])
        self._tool_manager = ToolManager(all_tools)
        self._tool_manager.add_provider(self.mcp)

        # Initialize conversation manager
        resources = list(resources)
        if ctx.config.knowledge:
            resources.extend(ctx.config.knowledge.get_resources())
        self.conversation = ConversationManager(self, memory_cfg, resources=resources)

        # Initialize provider
        match provider:
            case "pydantic_ai":
                from llmling_agent_providers.pydanticai import PydanticAIProvider

                if model and not isinstance(model, str):
                    from pydantic_ai import models

                    assert isinstance(model, models.Model)
                self._provider: AgentProvider = PydanticAIProvider(
                    model=model,
                    retries=retries,
                    end_strategy=end_strategy,
                    result_retries=result_retries,
                    defer_model_check=defer_model_check,
                    debug=debug,
                )
            case "human":
                from llmling_agent_providers.human import HumanProvider

                self._provider = HumanProvider(name=name, debug=debug)
            case Callable():
                from llmling_agent_providers.callback import CallbackProvider

                self._provider = CallbackProvider(provider, name=name, debug=debug)
            case "litellm":
                from llmling_agent_providers.litellm_provider import LiteLLMProvider

                self._provider = LiteLLMProvider(name=name, debug=debug, retries=retries)
            case AgentProvider():
                self._provider = provider
            case _:
                msg = f"Invalid agent type: {type}"
                raise ValueError(msg)
        self._provider.tool_manager = self._tool_manager
        self._provider.context = ctx
        self._provider.conversation = self.conversation
        ctx.capabilities.register_capability_tools(self)

        # init variables
        self._debug = debug
        self._result_type = None
        self.parallel_init = parallel_init
        self.name = name
        self._background_task: asyncio.Task[Any] | None = None

        # Forward provider signals
        self._provider.chunk_streamed.connect(self.chunk_streamed.emit)
        self._provider.model_changed.connect(self.model_changed.emit)
        self._provider.tool_used.connect(self.tool_used.emit)
        self._provider.model_changed.connect(self.model_changed.emit)

        self.talk = Interactions(self)
        self._logger = AgentLogger(self, enable_db_logging=memory_cfg.enable)

        # Set up system prompts
        config_prompts = ctx.config.system_prompts if ctx else []
        all_prompts: list[AnyPromptType] = list(config_prompts)
        if not isinstance(system_prompt, list):
            all_prompts.append(system_prompt)
        else:
            all_prompts.extend(system_prompt)
        self.sys_prompts = SystemPrompts(all_prompts, context=ctx)

    def __repr__(self) -> str:
        desc = f", {self.description!r}" if self.description else ""
        tools = f", tools={len(self.tools)}" if self.tools else ""
        return f"Agent({self._provider!r}{desc}{tools})"

    def __prompt__(self) -> str:
        typ = self._provider.__class__.__name__
        model = self.model_name or "default"
        parts = [f"Agent: {self.name}", f"Type: {typ}", f"Model: {model}"]
        if self.description:
            parts.append(f"Description: {self.description}")
        parts.extend([self.tools.__prompt__(), self.conversation.__prompt__()])

        return "\n".join(parts)

    async def __aenter__(self) -> Self:
        """Enter async context and set up MCP servers."""
        try:
            # Collect all coroutines that need to be run
            coros: list[Coroutine[Any, Any, Any]] = []

            # Runtime initialization if needed
            runtime_ref = self.context.runtime
            if runtime_ref and not runtime_ref._initialized:
                self._owns_runtime = True
                coros.append(runtime_ref.__aenter__())

            # Events initialization
            coros.append(super().__aenter__())

            # Get conversation init tasks directly
            coros.extend(self.conversation.get_initialization_tasks())

            # Execute coroutines either in parallel or sequentially
            if self.parallel_init and coros:
                await asyncio.gather(*coros)
            else:
                for coro in coros:
                    await coro
            if runtime_ref:
                runtime_tools = runtime_ref.tools.values()
                names = [t.name for t in runtime_tools]
                logger.debug("Registering runtime tools: %s", names)
                for tool in runtime_tools:
                    self.tools.register_tool(tool, source="runtime")
        except Exception as e:
            # Clean up in reverse order
            if self._owns_runtime and runtime_ref and self.context.runtime == runtime_ref:
                await runtime_ref.__aexit__(type(e), e, e.__traceback__)
            msg = "Failed to initialize agent"
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
        await super().__aexit__(exc_type, exc_val, exc_tb)
        try:
            await self.mcp.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if self._owns_runtime and self.context.runtime:
                await self.context.runtime.__aexit__(exc_type, exc_val, exc_tb)

    @overload
    def __and__(
        self, other: Agent[TDeps] | StructuredAgent[TDeps, Any]
    ) -> Team[TDeps]: ...

    @overload
    def __and__(self, other: Team[TDeps]) -> Team[TDeps]: ...

    @overload
    def __and__(self, other: ProcessorCallback[Any]) -> Team[TDeps]: ...

    def __and__(self, other: MessageNode[Any, Any] | ProcessorCallback[Any]) -> Team[Any]:
        """Create agent group using | operator.

        Example:
            group = analyzer & planner & executor  # Create group of 3
            group = analyzer & existing_group  # Add to existing group
        """
        from llmling_agent.agent import StructuredAgent
        from llmling_agent.delegation.team import Team

        match other:
            case Team():
                return Team([self, *other.agents])
            case Callable():
                if callable(other):
                    if has_return_type(other, str):
                        agent_2 = Agent.from_callback(other)
                    else:
                        agent_2 = StructuredAgent.from_callback(other)
                return Team([self, agent_2])
            case MessageNode():
                return Team([self, other])
            case _:
                msg = f"Invalid agent type: {type(other)}"
                raise ValueError(msg)

    @overload
    def __or__(self, other: MessageNode[TDeps, Any]) -> TeamRun[TDeps, Any]: ...

    @overload
    def __or__[TOtherDeps](
        self,
        other: MessageNode[TOtherDeps, Any],
    ) -> TeamRun[Any, Any]: ...

    @overload
    def __or__(self, other: ProcessorCallback[Any]) -> TeamRun[Any, Any]: ...

    def __or__(self, other: MessageNode[Any, Any] | ProcessorCallback[Any]) -> TeamRun:
        # Create new execution with sequential mode (for piping)
        from llmling_agent import StructuredAgent, TeamRun

        if callable(other):
            if has_return_type(other, str):
                other = Agent.from_callback(other)
            else:
                other = StructuredAgent.from_callback(other)

        return TeamRun([self, other])

    @classmethod
    def from_callback(
        cls,
        callback: ProcessorCallback[str],
        *,
        name: str | None = None,
        deps: TDeps | None = None,
        debug: bool = False,
        **kwargs: Any,
    ) -> Agent[TDeps]:
        """Create an agent from a processing callback.

        Args:
            callback: Function to process messages. Can be:
                - sync or async
                - with or without context
                - must return str for pipeline compatibility
            name: Optional name for the agent
            deps: Optional dependencies for the agent
            debug: Whether to enable debug mode
            kwargs: Additional arguments for agent
        """
        from llmling_agent_providers.callback import CallbackProvider

        name = name or callback.__name__ or "processor"
        provider = CallbackProvider[Any](callback, name=name)
        agent = cls(provider=provider, name=name, debug=debug, **kwargs)
        if deps is not None:
            agent.context.data = deps
        return agent

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._provider.name or "llmling-agent"

    @name.setter
    def name(self, value: str):
        self._provider.name = value

    @property
    def context(self) -> AgentContext[TDeps]:
        """Get agent context."""
        return self._provider.context

    @context.setter
    def context(self, value: AgentContext[TDeps]):
        """Set agent context and propagate to provider."""
        self._provider.context = value
        self.mcp.context = value

    def set_result_type(
        self,
        result_type: type[TResult] | str | ResponseDefinition | None,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ):
        """Set or update the result type for this agent.

        Args:
            result_type: New result type, can be:
                - A Python type for validation
                - Name of a response definition
                - Response definition instance
                - None to reset to unstructured mode
            tool_name: Optional override for tool name
            tool_description: Optional override for tool description
        """
        logger.debug("Setting result type to: %s for %r", result_type, self.name)
        self._result_type = to_type(result_type)  # to_type?

    @property
    def provider(self) -> AgentProvider:
        """Get the underlying provider."""
        return self._provider

    @provider.setter
    def provider(self, value: AgentProvider, model: ModelType = None):
        """Set the underlying provider."""
        from llmling_agent_providers.base import AgentProvider

        name = self.name
        debug = self._debug
        match value:
            case AgentProvider():
                self._provider = value
            case "pydantic_ai":
                from llmling_agent_providers.pydanticai import PydanticAIProvider

                self._provider = PydanticAIProvider(model=model, name=name, debug=debug)
            case "human":
                from llmling_agent_providers.human import HumanProvider

                self._provider = HumanProvider(name=name, debug=debug)
            case "litellm":
                from llmling_agent_providers.litellm_provider import LiteLLMProvider

                self._provider = LiteLLMProvider(model=model, name=name, debug=debug)
            case Callable():
                from llmling_agent_providers.callback import CallbackProvider

                self._provider = CallbackProvider(value, name=name, debug=debug)
            case _:
                msg = f"Invalid agent type: {type}"
                raise ValueError(msg)

    @overload
    def to_structured(
        self,
        result_type: None,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> Self: ...

    @overload
    def to_structured[TResult](
        self,
        result_type: type[TResult] | str | ResponseDefinition,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> StructuredAgent[TDeps, TResult]: ...

    def to_structured[TResult](
        self,
        result_type: type[TResult] | str | ResponseDefinition | None,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> StructuredAgent[TDeps, TResult] | Self:
        """Convert this agent to a structured agent.

        If result_type is None, returns self unchanged (no wrapping).
        Otherwise creates a StructuredAgent wrapper.

        Args:
            result_type: Type for structured responses. Can be:
                - A Python type (Pydantic model)
                - Name of response definition from context
                - Complete response definition
                - None to skip wrapping
            tool_name: Optional override for result tool name
            tool_description: Optional override for result tool description

        Returns:
            Either StructuredAgent wrapper or self unchanged
        from llmling_agent.agent import StructuredAgent
        """
        if result_type is None:
            return self

        from llmling_agent.agent import StructuredAgent

        return StructuredAgent(
            self,
            result_type=result_type,
            tool_name=tool_name,
            tool_description=tool_description,
        )

    @classmethod
    @overload
    def open(
        cls,
        config_path: StrPath | Config | None = None,
        *,
        result_type: None = None,
        model: ModelType = None,
        session: SessionIdType | SessionQuery = None,
        system_prompt: str | Sequence[str] = (),
        name: str = "llmling-agent",
        retries: int = 1,
        result_retries: int | None = None,
        end_strategy: EndStrategy = "early",
        debug: bool = False,
    ) -> AbstractAsyncContextManager[Agent[TDeps]]: ...

    @classmethod
    @overload
    def open[TResult](
        cls,
        config_path: StrPath | Config | None = None,
        *,
        result_type: type[TResult],
        model: ModelType = None,
        session: SessionIdType | SessionQuery = None,
        system_prompt: str | Sequence[str] = (),
        name: str = "llmling-agent",
        retries: int = 1,
        result_retries: int | None = None,
        end_strategy: EndStrategy = "early",
        debug: bool = False,
    ) -> AbstractAsyncContextManager[StructuredAgent[TDeps, TResult]]: ...

    @classmethod
    @asynccontextmanager
    async def open[TResult](
        cls,
        config_path: StrPath | Config | None = None,
        *,
        result_type: type[TResult] | None = None,
        model: ModelType = None,
        session: SessionIdType | SessionQuery = None,
        system_prompt: str | Sequence[str] = (),
        name: str = "llmling-agent",
        retries: int = 1,
        result_retries: int | None = None,
        end_strategy: EndStrategy = "early",
        debug: bool = False,
    ) -> AsyncIterator[Agent[TDeps] | StructuredAgent[TDeps, TResult]]:
        """Open and configure an agent with an auto-managed runtime configuration.

        Args:
            config_path: Path to the runtime configuration file or a Config instance
                        (defaults to Config())
            result_type: Optional type for structured responses
            model: The default model to use (defaults to GPT-4)
            session: Optional id or Session query to recover a conversation
            system_prompt: Static system prompts to use for this agent
            name: Name of the agent for logging
            retries: Default number of retries for failed operations
            result_retries: Max retries for result validation (defaults to retries)
            end_strategy: Strategy for handling tool calls that are requested alongside
                        a final result
            debug: Whether to enable debug logging

        Yields:
            Configured Agent instance

        Example:
            ```python
            async with Agent.open("config.yml") as agent:
                result = await agent.run("Hello!")
                print(result.data)
            ```
        """
        agent = cls(
            runtime=config_path,
            model=model,
            session=session,
            system_prompt=system_prompt,
            name=name,
            retries=retries,
            end_strategy=end_strategy,
            result_retries=result_retries,
            debug=debug,
        )
        async with agent:
            yield (agent if result_type is None else agent.to_structured(result_type))

    @classmethod
    @overload
    def open_agent(
        cls,
        config: StrPath | AgentsManifest,
        agent_name: str,
        *,
        deps: TDeps | None = None,
        result_type: None = None,
    ) -> AbstractAsyncContextManager[Agent[TDeps]]: ...

    @classmethod
    @overload
    def open_agent[TResult](
        cls,
        config: StrPath | AgentsManifest,
        agent_name: str,
        *,
        deps: TDeps | None = None,
        result_type: type[TResult],
    ) -> AbstractAsyncContextManager[StructuredAgent[TDeps, TResult]]: ...

    @classmethod
    @asynccontextmanager
    async def open_agent[TResult](
        cls,
        config: StrPath | AgentsManifest,
        agent_name: str,
        *,
        deps: TDeps | None = None,  # TDeps from class
        result_type: type[TResult] | None = None,
    ) -> AsyncIterator[Agent[TDeps] | StructuredAgent[TDeps, TResult]]:
        """Open and configure a specific agent from configuration."""
        """Implementation with all parameters..."""
        """Open and configure a specific agent from configuration.

        Args:
            config: Path to agent configuration file or AgentsManifest instance
            agent_name: Name of the agent to load

            # Basic Configuration
            deps: Optional deps for the agent
            result_type: Optional type for structured responses

        Yields:
            Configured Agent instance

        Raises:
            ValueError: If agent not found or configuration invalid
            RuntimeError: If agent initialization fails

        Example:
            ```python
            async with Agent.open_agent(
                "agents.yml",
                "my_agent",
                model="gpt-4",
                tools=[my_custom_tool],
            ) as agent:
                result = await agent.run("Do something")
            ```
        """
        from llmling_agent.agent import AgentContext

        if isinstance(config, AgentsManifest):
            agent_def = config
        else:
            agent_def = AgentsManifest.from_file(config)

        if agent_name not in agent_def.agents:
            msg = f"Agent {agent_name!r} not found in {config}"
            raise ValueError(msg)

        agent_config = agent_def.agents[agent_name]
        resolved_type = result_type or agent_def.get_result_type(agent_name)

        # Use model from override or agent config
        if not agent_config.model:
            msg = "Model must be specified in config"
            raise ValueError(msg)

        # Create context
        context = AgentContext[TDeps](
            node_name=agent_name,
            capabilities=agent_config.capabilities,
            definition=agent_def,
            config=agent_config,
        )

        # Set up runtime
        cfg = agent_config.get_config()
        base_agent = cls(runtime=cfg, context=context, model=agent_config.model)
        async with base_agent:
            if resolved_type is not None and resolved_type is not str:
                # Yield structured agent with correct typing
                from llmling_agent.agent.structured import StructuredAgent

                yield StructuredAgent[TDeps, TResult](base_agent, resolved_type)
            else:
                yield base_agent

    def is_busy(self) -> bool:
        """Check if agent is currently processing tasks."""
        return bool(self._pending_tasks or self._background_task)

    @property
    def model_name(self) -> str | None:
        """Get the model name in a consistent format."""
        return self._provider.model_name

    @logfire.instrument("Calling Agent.run: {prompts}:")
    async def _run(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | ChatMessage[Any],
        result_type: type[TResult] | None = None,
        model: ModelType = None,
        store_history: bool = True,
        tool_choice: bool | str | list[str] = True,
        wait_for_connections: bool | None = None,
    ) -> ChatMessage[TResult]:
        """Run agent with prompt and get response.

        Args:
            prompts: User query or instruction
            result_type: Optional type for structured responses
            model: Optional model override
            store_history: Whether the message exchange should be added to the
                           context window
            tool_choice: Control tool usage:
                - True: Allow all tools
                - False: No tools
                - str: Use specific tool
                - list[str]: Allow specific tools
            wait_for_connections: Whether to wait for connected agents to complete

        Returns:
            Result containing response and run information

        Raises:
            UnexpectedModelBehavior: If the model fails or behaves unexpectedly
        """
        """Run agent with prompt and get response."""
        message_id = str(uuid4())

        tools = await self.tools.get_tools(state="enabled")
        match tool_choice:
            case str():
                tools = [t for t in tools if t.name == tool_choice]
            case list():
                tools = [t for t in tools if t.name in tool_choice]
            case False:
                tools = []
            case True | None:
                pass  # Keep all tools
        self.set_result_type(result_type)
        start_time = time.perf_counter()
        sys_prompt = await self.sys_prompts.format_system_prompt(self)
        message_history = self.conversation.get_history()
        try:
            result = await self._provider.generate_response(
                *await convert_prompts(prompts),
                message_id=message_id,
                message_history=message_history,
                tools=tools,
                result_type=result_type,
                model=model,
                system_prompt=sys_prompt,
            )
        except Exception as e:
            logger.exception("Agent run failed")
            self.run_failed.emit("Agent run failed", e)
            raise
        else:
            response_msg = ChatMessage[TResult](
                content=result.content,
                role="assistant",
                name=self.name,
                model=result.model_name,
                message_id=message_id,
                tool_calls=result.tool_calls,
                cost_info=result.cost_and_usage,
                response_time=time.perf_counter() - start_time,
                provider_extra=result.provider_extra or {},
            )
            if self._debug:
                import devtools

                devtools.debug(response_msg)
            return response_msg

    def to_agent_tool(
        self,
        *,
        name: str | None = None,
        reset_history_on_run: bool = True,
        pass_message_history: bool = False,
        share_context: bool = False,
        parent: AnyAgent[Any, Any] | None = None,
    ) -> LLMCallableTool:
        """Create a tool from this agent.

        Args:
            name: Optional tool name override
            reset_history_on_run: Clear agent's history before each run
            pass_message_history: Pass parent's message history to agent
            share_context: Whether to pass parent's context/deps
            parent: Optional parent agent for history/context sharing
        """
        tool_name = f"ask_{self.name}"

        async def wrapped_tool(prompt: str) -> str:
            if pass_message_history and not parent:
                msg = "Parent agent required for message history sharing"
                raise ToolError(msg)

            if reset_history_on_run:
                self.conversation.clear()

            history = None
            if pass_message_history and parent:
                history = parent.conversation.get_history()
                old = self.conversation.get_history()
                self.conversation.set_history(history)
            result = await self.run(prompt, result_type=self._result_type)
            if history:
                self.conversation.set_history(old)
            return result.data

        normalized_name = self.name.replace("_", " ").title()
        docstring = f"Get expert answer from specialized agent: {normalized_name}"
        if self.description:
            docstring = f"{docstring}\n\n{self.description}"

        wrapped_tool.__doc__ = docstring
        wrapped_tool.__name__ = tool_name

        return LLMCallableTool.from_callable(
            wrapped_tool,
            name_override=tool_name,
            description_override=docstring,
        )

    @asynccontextmanager
    async def run_stream(
        self,
        *prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
        result_type: type[TResult] | None = None,
        model: ModelType = None,
        tool_choice: bool | str | list[str] = True,
        store_history: bool = True,
        wait_for_connections: bool | None = None,
    ) -> AsyncIterator[StreamingResponseProtocol[TResult]]:
        """Run agent with prompt and get a streaming response.

        Args:
            prompt: User query or instruction
            result_type: Optional type for structured responses
            model: Optional model override
            tool_choice: Control tool usage:
                - True: Allow all tools
                - False: No tools
                - str: Use specific tool
                - list[str]: Allow specific tools
            store_history: Whether the message exchange should be added to the
                           context window
            wait_for_connections: Whether to wait for connected agents to complete

        Returns:
            A streaming result to iterate over.

        Raises:
            UnexpectedModelBehavior: If the model fails or behaves unexpectedly
        """
        prompts = await convert_prompts(prompt)
        final_prompt = "\n\n".join(str(p) for p in prompts)
        self.set_result_type(result_type)
        self.context.current_prompt = final_prompt
        # Create and emit user message
        user_msg = ChatMessage[str](content=final_prompt, role="user")
        self.message_received.emit(user_msg)
        message_id = str(uuid4())
        start_time = time.perf_counter()
        sys_prompt = await self.sys_prompts.format_system_prompt(self)
        tools = await self.tools.get_tools(state="enabled")
        match tool_choice:
            case str():
                tools = [t for t in tools if t.name == tool_choice]
            case list():
                tools = [t for t in tools if t.name in tool_choice]
            case False:
                tools = []
            case True | None:
                pass  # Keep all tools
        try:
            message_history = self.conversation.get_history()
            async with self._provider.stream_response(
                *prompts,
                message_id=message_id,
                message_history=message_history,
                result_type=result_type,
                model=model,
                store_history=store_history,
                tools=tools,
                system_prompt=sys_prompt,
            ) as stream:
                yield stream
                usage = stream.usage()
                cost_info = None
                model_name = stream.model_name  # type: ignore
                if model_name:
                    cost_info = await TokenCost.from_usage(
                        usage,
                        model_name,
                        final_prompt,
                        str(stream.formatted_content),  # type: ignore
                    )
                response_msg = ChatMessage[TResult](
                    content=cast(TResult, stream.formatted_content),  # type: ignore
                    role="assistant",
                    name=self.name,
                    model=model_name,
                    message_id=message_id,
                    cost_info=cost_info,
                    response_time=time.perf_counter() - start_time,
                    # provider_extra=stream.provider_extra or {},
                )
                self.message_sent.emit(response_msg)
                await self.connections.route_message(
                    response_msg,
                    wait=wait_for_connections,
                )

        except Exception as e:
            logger.exception("Agent stream failed")
            self.run_failed.emit("Agent stream failed", e)
            raise

    async def run_iter(
        self,
        *prompt_groups: Sequence[AnyPromptType | PIL.Image.Image | os.PathLike[str]],
        result_type: type[TResult] | None = None,
        model: ModelType = None,
        store_history: bool = True,
        wait_for_connections: bool | None = None,
    ) -> AsyncIterator[ChatMessage[TResult]]:
        """Run agent sequentially on multiple prompt groups.

        Args:
            prompt_groups: Groups of prompts to process sequentially
            result_type: Optional type for structured responses
            model: Optional model override
            store_history: Whether to store in conversation history
            wait_for_connections: Whether to wait for connected agents

        Yields:
            Response messages in sequence

        Example:
            questions = [
                ["What is your name?"],
                ["How old are you?", image1],
                ["Describe this image", image2],
            ]
            async for response in agent.run_iter(*questions):
                print(response.content)
        """
        for prompts in prompt_groups:
            response = await self.run(
                *prompts,
                result_type=result_type,
                model=model,
                store_history=store_history,
                wait_for_connections=wait_for_connections,
            )
            yield response  # pyright: ignore

    def run_sync(
        self,
        *prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
        result_type: type[TResult] | None = None,
        deps: TDeps | None = None,
        model: ModelType = None,
        store_history: bool = True,
    ) -> ChatMessage[TResult]:
        """Run agent synchronously (convenience wrapper).

        Args:
            prompt: User query or instruction
            result_type: Optional type for structured responses
            deps: Optional dependencies for the agent
            model: Optional model override
            store_history: Whether the message exchange should be added to the
                           context window
        Returns:
            Result containing response and run information
        """
        coro = self.run(
            *prompt,
            model=model,
            store_history=store_history,
            result_type=result_type,
        )
        return self.run_task_sync(coro)  # type: ignore

    async def run_job(
        self,
        job: Job[TDeps, str | None],
        *,
        store_history: bool = True,
        include_agent_tools: bool = True,
    ) -> ChatMessage[str]:
        """Execute a pre-defined task.

        Args:
            job: Job configuration to execute
            store_history: Whether the message exchange should be added to the
                           context window
            include_agent_tools: Whether to include agent tools
        Returns:
            Job execution result

        Raises:
            JobError: If task execution fails
            ValueError: If task configuration is invalid
        """
        from llmling_agent.tasks import JobError

        if job.required_dependency is not None:  # noqa: SIM102
            if not isinstance(self.context.data, job.required_dependency):
                msg = (
                    f"Agent dependencies ({type(self.context.data)}) "
                    f"don't match job requirement ({job.required_dependency})"
                )
                raise JobError(msg)

        # Load task knowledge
        if job.knowledge:
            # Add knowledge sources to context
            resources: list[Resource | str] = list(job.knowledge.paths) + list(
                job.knowledge.resources
            )
            for source in resources:
                await self.conversation.load_context_source(source)
            for prompt in job.knowledge.prompts:
                await self.conversation.load_context_source(prompt)
        try:
            # Register task tools temporarily
            tools = job.get_tools()
            with self.tools.temporary_tools(tools, exclusive=not include_agent_tools):
                # Execute job with job-specific tools
                return await self.run(await job.get_prompt(), store_history=store_history)

        except Exception as e:
            msg = f"Task execution failed: {e}"
            logger.exception(msg)
            raise JobError(msg) from e

    async def run_in_background(
        self,
        *prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
        max_count: int | None = None,
        interval: float = 1.0,
        block: bool = False,
        **kwargs: Any,
    ) -> ChatMessage[TResult] | None:
        """Run agent continuously in background with prompt or dynamic prompt function.

        Args:
            prompt: Static prompt or function that generates prompts
            max_count: Maximum number of runs (None = infinite)
            interval: Seconds between runs
            block: Whether to block until completion
            **kwargs: Arguments passed to run()
        """

        async def _continuous():
            count = 0
            msg = "%s: Starting continuous run (max_count=%s, interval=%s) for %r"
            logger.debug(msg, self.name, max_count, interval, self.name)
            while max_count is None or count < max_count:
                try:
                    current_prompts = [
                        call_with_context(p, self.context, **kwargs) if callable(p) else p
                        for p in prompt
                    ]
                    msg = "%s: Generated prompt #%d: %s"
                    logger.debug(msg, self.name, count, current_prompts)

                    await self.run(current_prompts, **kwargs)
                    msg = "%s: Run continous result #%d"
                    logger.debug(msg, self.name, count)

                    count += 1
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    logger.debug("%s: Continuous run cancelled", self.name)
                    break
                except Exception:
                    logger.exception("%s: Background run failed", self.name)
                    await asyncio.sleep(interval)
            msg = "%s: Continuous run completed after %d iterations"
            logger.debug(msg, self.name, count)

        # Cancel any existing background task
        await self.stop()
        task = asyncio.create_task(_continuous(), name=f"background_{self.name}")

        if block:
            try:
                await task
                return None
            finally:
                if not task.done():
                    task.cancel()
        else:
            logger.debug("%s: Started background task %s", self.name, task.get_name())
            self._background_task = task
            return None

    async def stop(self):
        """Stop continuous execution if running."""
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            await self._background_task
            self._background_task = None

    async def wait(self) -> ChatMessage[TResult]:
        """Wait for background execution to complete."""
        if not self._background_task:
            msg = "No background task running"
            raise RuntimeError(msg)
        if self._infinite:
            msg = "Cannot wait on infinite execution"
            raise RuntimeError(msg)
        try:
            return await self._background_task
        finally:
            self._background_task = None

    def clear_history(self):
        """Clear both internal and pydantic-ai history."""
        self._logger.clear_state()
        self.conversation.clear()
        logger.debug("Cleared history and reset tool state")

    async def share(
        self,
        target: AnyAgent[TDeps, Any],
        *,
        tools: list[str] | None = None,
        resources: list[str] | None = None,
        history: bool | int | None = None,  # bool or number of messages
        token_limit: int | None = None,
    ):
        """Share capabilities and knowledge with another agent.

        Args:
            target: Agent to share with
            tools: List of tool names to share
            resources: List of resource names to share
            history: Share conversation history:
                    - True: Share full history
                    - int: Number of most recent messages to share
                    - None: Don't share history
            token_limit: Optional max tokens for history

        Raises:
            ValueError: If requested items don't exist
            RuntimeError: If runtime not available for resources
        """
        # Share tools if requested
        for name in tools or []:
            if tool := self.tools.get(name):
                meta = {"shared_from": self.name}
                target.tools.register_tool(tool.callable, metadata=meta)
            else:
                msg = f"Tool not found: {name}"
                raise ValueError(msg)

        # Share resources if requested
        if resources:
            if not self.runtime:
                msg = "No runtime available for sharing resources"
                raise RuntimeError(msg)
            for name in resources:
                if resource := self.runtime.get_resource(name):
                    await target.conversation.load_context_source(resource)  # type: ignore
                else:
                    msg = f"Resource not found: {name}"
                    raise ValueError(msg)

        # Share history if requested
        if history:
            history_text = await self.conversation.format_history(
                max_tokens=token_limit,
                num_messages=history if isinstance(history, int) else None,
            )
            target.conversation.add_context_message(
                history_text, source=self.name, metadata={"type": "shared_history"}
            )

    def register_worker(
        self,
        worker: AnyAgent[Any, Any],
        *,
        name: str | None = None,
        reset_history_on_run: bool = True,
        pass_message_history: bool = False,
        share_context: bool = False,
    ) -> ToolInfo:
        """Register another agent as a worker tool."""
        return self.tools.register_worker(
            worker,
            name=name,
            reset_history_on_run=reset_history_on_run,
            pass_message_history=pass_message_history,
            share_context=share_context,
            parent=self if (pass_message_history or share_context) else None,
        )

    def set_model(self, model: ModelType):
        """Set the model for this agent.

        Args:
            model: New model to use (name or instance)

        Emits:
            model_changed signal with the new model
        """
        self._provider.set_model(model)

    async def reset(self):
        """Reset agent state (conversation history and tool states)."""
        old_tools = await self.tools.list_tools()
        self.conversation.clear()
        self.tools.reset_states()
        new_tools = await self.tools.list_tools()

        event = self.AgentReset(
            agent_name=self.name,
            previous_tools=old_tools,
            new_tools=new_tools,
        )
        self.agent_reset.emit(event)

    @property
    def runtime(self) -> RuntimeConfig:
        """Get runtime configuration from context."""
        assert self.context.runtime
        return self.context.runtime

    @runtime.setter
    def runtime(self, value: RuntimeConfig):
        """Set runtime configuration and update context."""
        self.context.runtime = value

    @property
    def tools(self) -> ToolManager:
        return self._tool_manager

    @property
    def stats(self) -> MessageStats:
        return MessageStats(messages=self._logger.message_history)

    @asynccontextmanager
    async def temporary_state(
        self,
        *,
        system_prompts: list[AnyPromptType] | None = None,
        replace_prompts: bool = False,
        tools: list[ToolType] | None = None,
        replace_tools: bool = False,
        history: list[AnyPromptType] | SessionQuery | None = None,
        replace_history: bool = False,
        pause_routing: bool = False,
        model: ModelType | None = None,
        provider: AgentProvider | None = None,
    ) -> AsyncIterator[Self]:
        """Temporarily modify agent state.

        Args:
            system_prompts: Temporary system prompts to use
            replace_prompts: Whether to replace existing prompts
            tools: Temporary tools to make available
            replace_tools: Whether to replace existing tools
            history: Conversation history (prompts or query)
            replace_history: Whether to replace existing history
            pause_routing: Whether to pause message routing
            model: Temporary model override
            provider: Temporary provider override
        """
        old_model = self._provider.model if hasattr(self._provider, "model") else None  # pyright: ignore
        old_provider = self._provider

        async with AsyncExitStack() as stack:
            # System prompts (async)
            if system_prompts is not None:
                await stack.enter_async_context(
                    self.sys_prompts.temporary_prompt(
                        system_prompts, exclusive=replace_prompts
                    )
                )

            # Tools (sync)
            if tools is not None:
                stack.enter_context(
                    self.tools.temporary_tools(tools, exclusive=replace_tools)
                )

            # History (async)
            if history is not None:
                await stack.enter_async_context(
                    self.conversation.temporary_state(
                        history, replace_history=replace_history
                    )
                )

            # Routing (async)
            if pause_routing:
                await stack.enter_async_context(self.connections.paused_routing())

            # Model/Provider
            if provider is not None:
                self._provider = provider
            elif model is not None:
                self._provider.set_model(model)

            try:
                yield self
            finally:
                # Restore model/provider
                if provider is not None:
                    self._provider = old_provider
                elif model is not None and old_model:
                    self._provider.set_model(old_model)


if __name__ == "__main__":
    # import logging

    from llmling_agent import config_resources

    # logging.basicConfig(level=logging.DEBUG)

    sys_prompt = "Open browser with google, please"
    path = config_resources.OPEN_BROWSER
    _model = "openai:gpt-4o-mini"

    async def main():
        async with Agent[None].open(path, model=_model, debug=True) as agent:
            print(agent.tools.list_items())
            result = await agent.run(sys_prompt)
            print(result.data)

    asyncio.run(main())
