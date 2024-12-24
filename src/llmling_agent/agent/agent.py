"""LLMling integration with PydanticAI for AI-powered resource interaction."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence  # noqa: TC003
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from llmling.config.runtime import RuntimeConfig
from psygnal import Signal
from pydantic_ai import Agent as PydanticAgent, capture_run_messages
from pydantic_ai.models import infer_model
from typing_extensions import TypeVar

from llmling_agent.log import get_logger
from llmling_agent.models import AgentContext, AgentsManifest
from llmling_agent.models.agents import ToolCallInfo
from llmling_agent.models.messages import ChatMessage, MessageMetadata
from llmling_agent.pydantic_ai_utils import (
    convert_model_message,
    extract_usage,
    get_tool_calls,
)
from llmling_agent.responses import InlineResponseDefinition, resolve_response_type
from llmling_agent.tools.manager import ToolManager


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    import os

    from llmling.tools import LLMCallableTool
    from pydantic_ai.agent import EndStrategy, models
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.result import RunResult, StreamedRunResult


logger = get_logger(__name__)

TResult = TypeVar("TResult", default=str)
TDeps = TypeVar("TDeps", default=Any)


JINJA_PROC = "jinja_template"  # Name of builtin LLMling Jinja2 processor


class LLMlingAgent[TDeps, TResult]:
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

    message_received = Signal(ChatMessage[str])  # Always string
    message_sent = Signal(ChatMessage[TResult])
    message_exchanged = Signal(ChatMessage[TResult | str])
    tool_used = Signal(ToolCallInfo)  # Now we emit the whole info object
    model_changed = Signal(object)  # Model | None
    chunk_streamed = Signal(str)
    # `outbox` defined in __init__
    outbox = Signal(object, ChatMessage[Any])

    def __init__(
        self,
        runtime: RuntimeConfig,
        context: AgentContext[TDeps] | None = None,
        result_type: type[TResult] | None = None,
        *,
        model: models.Model | models.KnownModelName | None = None,
        system_prompt: str | Sequence[str] = (),
        name: str = "llmling-agent",
        tools: Sequence[LLMCallableTool] = (),
        retries: int = 1,
        result_tool_name: str = "final_result",
        result_tool_description: str | None = None,
        result_retries: int | None = None,
        tool_choice: bool | str | list[str] = True,
        end_strategy: EndStrategy = "early",
        defer_model_check: bool = False,
        enable_logging: bool = True,
        **kwargs,
    ):
        """Initialize agent with runtime configuration.

        Args:
            runtime: Runtime configuration providing access to resources/tools
            context: Agent context with capabilities and configuration
            result_type: Optional type for structured responses
            model: The default model to use (defaults to GPT-4)
            system_prompt: Static system prompts to use for this agent
            name: Name of the agent for logging
            tools: List of tools to register with the agent
            retries: Default number of retries for failed operations
            result_tool_name: Name of the tool used for final result
            result_tool_description: Description of the final result tool
            result_retries: Max retries for result validation (defaults to retries)
            tool_choice: Ability to set a fixed tool or temporarily disable tools usage.
            end_strategy: Strategy for handling tool calls that are requested alongside
                          a final result
            defer_model_check: Whether to defer model evaluation until first run
            kwargs: Additional arguments for PydanticAI agent
            enable_logging: Whether to enable logging for the agent
        """
        self._runtime = runtime
        self._context = context or AgentContext[TDeps].create_default(name)
        self._context.runtime = runtime

        self.message_received.connect(self.message_exchanged.emit)
        self.message_sent.connect(self.message_exchanged.emit)
        # self.outbox = Signal(LLMlingAgent[Any, Any], ChatMessage[Any])
        self.message_sent.connect(self._forward_message)

        # Initialize tool manager
        all_tools = list(tools)
        all_tools.extend(runtime.tools.values())  # Add runtime tools directly
        self._tool_manager = ToolManager(tools=all_tools, tool_choice=tool_choice)
        self._tool_manager.setup_history_tools(self._context.capabilities)

        # Resolve result type
        actual_type: type[TResult]
        match result_type:
            case str():
                actual_type = resolve_response_type(result_type, context)  # type: ignore[assignment]
            case InlineResponseDefinition():
                actual_type = resolve_response_type(result_type, None)  # type: ignore[assignment]
            case None | type():
                actual_type = result_type or str  # type: ignore[assignment]
            case _:
                msg = f"Invalid result_type: {type(result_type)}"
                raise TypeError(msg)
        # Initialize agent with all tools
        self._pydantic_agent = PydanticAgent(
            model=model,
            result_type=actual_type,
            system_prompt=system_prompt,
            deps_type=AgentContext,
            tools=[],  # tools get added for each call explicitely
            retries=retries,
            result_tool_name=result_tool_name,
            result_tool_description=result_tool_description,
            end_strategy=end_strategy,
            result_retries=result_retries,
            defer_model_check=defer_model_check,
            **kwargs,
        )
        self._name = name
        msg = "Initialized %s (model=%s, result_type=%s)"
        logger.debug(msg, self._name, model, result_type or "str")

        from llmling_agent.agent import AgentLogger
        from llmling_agent.events import EventManager

        self._logger = AgentLogger(self, enable_logging=enable_logging)
        self._events = EventManager(self, enable_events=True)
        self._pending_tasks: set[asyncio.Task[Any]] = set()
        self._connected_agents: set[LLMlingAgent[Any, Any]] = set()
        self._last_messages: list[ModelMessage] = []

    @classmethod
    @asynccontextmanager
    async def open(
        cls,
        config_path: str | os.PathLike[str],
        result_type: type[TResult] | None = None,
        *,
        model: models.Model | models.KnownModelName | None = None,
        system_prompt: str | Sequence[str] = (),
        name: str = "llmling-agent",
        retries: int = 1,
        result_tool_name: str = "final_result",
        result_tool_description: str | None = None,
        result_retries: int | None = None,
        end_strategy: EndStrategy = "early",
        defer_model_check: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[LLMlingAgent[TDeps, TResult]]:
        """Create an agent with an auto-managed runtime configuration.

        This is a convenience method that combines RuntimeConfig.open with agent creation.

        Args:
            config_path: Path to the runtime configuration file
            result_type: Optional type for structured responses
            model: The default model to use (defaults to GPT-4)
            system_prompt: Static system prompts to use for this agent
            name: Name of the agent for logging
            retries: Default number of retries for failed operations
            result_tool_name: Name of the tool used for final result
            result_tool_description: Description of the final result tool
            result_retries: Max retries for result validation (defaults to retries)
            end_strategy: Strategy for handling tool calls that are requested alongside
                          a final result
            defer_model_check: Whether to defer model evaluation until first run
            **kwargs: Additional arguments for PydanticAI agent

        Yields:
            Configured LLMlingAgent instance

        Example:
            ```python
            async with LLMlingAgent.open("config.yml") as agent:
                result = await agent.run("Hello!")
                print(result.data)
            ```
        """
        async with RuntimeConfig.open(config_path) as runtime:
            agent = cls(
                runtime=runtime,
                result_type=result_type,
                model=model,
                system_prompt=system_prompt,
                name=name,
                retries=retries,
                end_strategy=end_strategy,
                result_tool_name=result_tool_name,
                result_tool_description=result_tool_description,
                result_retries=result_retries,
                defer_model_check=defer_model_check,
                **kwargs,
            )
            try:
                yield agent
            finally:
                # Any cleanup if needed
                pass

    @classmethod
    @asynccontextmanager
    async def open_agent(
        cls,
        config: str | os.PathLike[str] | AgentsManifest,
        agent_name: str,
        *,
        # Model configuration
        model: str | models.Model | models.KnownModelName | None = None,
        result_type: type[TResult] | None = None,
        model_settings: dict[str, Any] | None = None,
        # Tool configuration
        tools: list[str | Callable[..., Any]] | None = None,
        tool_choice: bool | str | list[str] = True,
        end_strategy: EndStrategy = "early",
        # Execution settings
        retries: int = 1,
        result_tool_name: str = "final_result",
        result_tool_description: str | None = None,
        result_retries: int | None = None,
        # Other settings
        system_prompt: str | Sequence[str] | None = None,
        enable_logging: bool = True,
    ) -> AsyncIterator[LLMlingAgent[TDeps, TResult]]:
        """Open and configure a specific agent from configuration.

        Args:
            config: Path to agent configuration file or AgentsManifest instance
            agent_name: Name of the agent to load

            # Model Configuration
            model: Optional model override
            result_type: Optional type for structured responses
            model_settings: Additional model-specific settings

            # Tool Configuration
            tools: Additional tools to register (import paths or callables)
            tool_choice: Control tool usage:
                - True: Allow all tools
                - False: No tools
                - str: Use specific tool
                - list[str]: Allow specific tools
            end_strategy: Strategy for handling tool calls that are requested alongside
                            a final result

            # Execution Settings
            retries: Default number of retries for failed operations
            result_tool_name: Name of the tool used for final result
            result_tool_description: Description of the final result tool
            result_retries: Max retries for result validation (defaults to retries)

            # Other Settings
            system_prompt: Additional system prompts
            enable_logging: Whether to enable logging for the agent

        Yields:
            Configured LLMlingAgent instance

        Raises:
            ValueError: If agent not found or configuration invalid
            RuntimeError: If agent initialization fails

        Example:
            ```python
            async with LLMlingAgent.open_agent(
                "agents.yml",
                "my_agent",
                model="gpt-4",
                tools=[my_custom_tool],
            ) as agent:
                result = await agent.run("Do something")
            ```
        """
        if isinstance(config, AgentsManifest):
            agent_def = config
        else:
            agent_def = AgentsManifest.from_file(config)

        if agent_name not in agent_def.agents:
            msg = f"Agent {agent_name!r} not found in {config}"
            raise ValueError(msg)

        agent_config = agent_def.agents[agent_name]

        # Use model from override or agent config
        actual_model = model or agent_config.model
        if not actual_model:
            msg = "Model must be specified either in config or as override"
            raise ValueError(msg)

        capabilities = agent_def.get_capabilities(agent_config.role)

        # Create context
        context: AgentContext[TDeps] = AgentContext(
            agent_name=agent_name,
            capabilities=capabilities,
            definition=agent_def,
            config=agent_config,
            model_settings=model_settings or {},
        )

        # Set up runtime
        cfg = agent_config.get_config()
        async with RuntimeConfig.open(cfg) as runtime:
            if tools:
                for tool in tools:
                    await runtime.register_tool(tool)

            agent = cls(
                runtime=runtime,
                context=context,
                result_type=result_type,
                model=actual_model,  # type: ignore[arg-type]
                retries=retries,
                result_tool_name=result_tool_name,
                result_tool_description=result_tool_description,
                result_retries=result_retries,
                end_strategy=end_strategy,
                tool_choice=tool_choice,
                system_prompt=system_prompt or [],
                enable_logging=enable_logging,
            )
            try:
                yield agent
            finally:
                # Any cleanup if needed
                pass

    def _forward_message(self, message: ChatMessage[Any]):
        """Forward sent messages."""
        logger.debug("forwarding message from %s: %s", self.name, message.content)
        self.outbox.emit(self, message)

    def pass_results_to(self, other: LLMlingAgent[Any, Any]):
        """Forward results to another agent."""
        self.outbox.connect(other._handle_message)
        self._connected_agents.add(other)

    def stop_passing_results_to(self, other: LLMlingAgent[Any, Any]):
        """Stop forwarding results to another agent."""
        if other in self._connected_agents:
            self.outbox.disconnect(other._handle_message)
            self._connected_agents.remove(other)

    @property
    def model_name(self) -> str | None:
        """Get the model name in a consistent format."""
        match self._pydantic_agent.model:
            case str() | None:
                return self._pydantic_agent.model
            case _:
                return self._pydantic_agent.model.name()

    def _update_tools(self):
        """Update pydantic-ai tools."""
        self._pydantic_agent._function_tools.clear()
        for tool in self.tools.get_tools(state="enabled"):
            assert tool._original_callable
            self._pydantic_agent.tool_plain(tool._original_callable)

    async def run(
        self,
        prompt: str,
        *,
        deps: TDeps | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        # wait_for_chain: bool = True,
    ) -> RunResult[TResult]:
        """Run agent with prompt and get response.

        Args:
            prompt: User query or instruction
            deps: Optional dependencies for the agent
            message_history: Optional previous messages for context
            model: Optional model override

        Returns:
            Result containing response and run information

        Raises:
            UnexpectedModelBehavior: If the model fails or behaves unexpectedly
        """
        wait_for_chain = False  # TODO
        if deps is not None:
            self._context.data = deps
        try:
            # Clear all tools
            if self._context:
                self._context.current_prompt = prompt
            if model:
                # perhaps also check for old model == new model?
                if isinstance(model, str):
                    model = infer_model(model)
                self.model_changed.emit(model)
            # Register currently enabled tools
            self._update_tools()

            logger.debug("agent run prompt=%s", prompt)

            with capture_run_messages() as messages:
                # Run through pydantic-ai's public interface
                result = await self._pydantic_agent.run(
                    prompt,
                    deps=self._context,
                    message_history=message_history,
                    model=model,
                )

                # Emit user message
                user_msg: ChatMessage[str] = ChatMessage(content=prompt, role="user")
                self.message_received.emit(user_msg)

                # Get cost info for assistant response
                result_str = str(result.data)
                usage = result.usage()
                cost = (
                    await extract_usage(usage, self.model_name, prompt, result_str)
                    if self.model_name
                    else None
                )
                message_id = str(uuid4())

                # Create and emit assistant message
                meta = MessageMetadata(
                    model=self.model_name,
                    token_usage=cost.token_usage if cost else None,
                    cost=cost.cost_usd if cost else None,
                )
                assistant_msg: ChatMessage[TResult] = ChatMessage[TResult](
                    content=result.data,
                    role="assistant",
                    message_id=message_id,
                    metadata=meta,
                )
                for call in get_tool_calls(messages):
                    call.message_id = message_id
                    call.context_data = self._context.data if self._context else None
                    self.tool_used.emit(call)
                self._last_messages = list(messages)
            self.message_sent.emit(assistant_msg)

        except Exception:
            logger.exception("Agent run failed")
            raise
        else:
            if wait_for_chain:
                await self.wait_for_chain()
            return result
        finally:
            if model:
                # Restore original model in signal
                old = self._pydantic_agent.model
                model_obj = infer_model(old) if isinstance(old, str) else old
                self.model_changed.emit(model_obj)

    async def run_stream(
        self,
        prompt: str,
        *,
        deps: TDeps | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        # wait_for_chain: bool = True,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentContext[TDeps], TResult]]:
        """Run agent with prompt and get streaming response.

        Args:
            prompt: User query or instruction
            deps: Optional dependencies for the agent
            message_history: Optional previous messages for context
            model: Optional model override

        Returns:
            Async context manager for streaming the response

        Example:
            ```python
            stream_ctx = agent.run_stream("Hello!")
            async with await stream_ctx as result:
                async for message in result.stream():
                    print(message)
            ```
        """
        wait_for_chain = False
        if deps is not None:
            self._context.data = deps
        try:
            self._update_tools()

            # Emit user message
            user_msg: ChatMessage[str] = ChatMessage(content=prompt, role="user")
            self.message_received.emit(user_msg)

            stream_ctx = self._pydantic_agent.run_stream(
                prompt,
                message_history=message_history,
                model=model,
                deps=self._context,
            )

            @asynccontextmanager
            async def wrapper():
                message_id = str(uuid4())
                with capture_run_messages() as messages:
                    async with stream_ctx as stream:
                        # Monkey patch the stream method to emit chunks
                        original_stream = stream.stream

                        async def patched_stream():
                            async for chunk in original_stream():
                                self.chunk_streamed.emit(str(chunk))
                                yield chunk

                        stream.stream = patched_stream  # type: ignore
                        yield stream

                        # After completion
                        if stream.is_complete:
                            result_str = str(await stream.get_data())
                            usage = stream.usage()
                            cost = (
                                await extract_usage(
                                    usage, self.model_name, prompt, result_str
                                )
                                if self.model_name
                                else None
                            )

                            # Handle tool calls after completion
                            for call in get_tool_calls(messages):
                                call.message_id = message_id
                                call.context_data = (
                                    self._context.data if self._context else None
                                )
                                self.tool_used.emit(call)

                            # Create and emit assistant message
                            meta = MessageMetadata(
                                model=self.model_name,
                                token_usage=cost.token_usage if cost else None,
                                cost=cost.cost_usd if cost else None,
                            )
                            assistant_msg: ChatMessage[TResult] = ChatMessage[TResult](
                                content=await stream.get_data(),
                                role="assistant",
                                message_id=message_id,
                                metadata=meta,
                            )
                            # Store history
                            self._last_messages = list(messages)
                self.message_sent.emit(assistant_msg)
                if wait_for_chain:
                    await self.wait_for_chain()

            return wrapper()

        except Exception:
            logger.exception("Agent stream failed")
            raise

    def run_sync(
        self,
        prompt: str,
        *,
        deps: TDeps | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> RunResult[TResult]:
        """Run agent synchronously (convenience wrapper).

        Args:
            prompt: User query or instruction
            deps: Optional dependencies for the agent
            message_history: Optional previous messages for context
            model: Optional model override

        Returns:
            Result containing response and run information
        """
        try:
            return asyncio.run(
                self.run(prompt, message_history=message_history, deps=deps, model=model)
            )
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.exception("Sync agent run failed")
            raise

    async def complete_tasks(self):
        """Wait for all pending tasks to complete."""
        if self._pending_tasks:
            await asyncio.wait(self._pending_tasks)

    async def wait_for_chain(self, _seen: set[str] | None = None):
        """Wait for this agent and all connected agents to complete their tasks."""
        # Track seen agents to avoid cycles
        seen = _seen or {self.name}

        # Wait for our own tasks
        await self.complete_tasks()

        # Wait for connected agents
        for agent in self._connected_agents:
            if agent.name not in seen:
                seen.add(agent.name)
                await agent.wait_for_chain(seen)

    def system_prompt(self, *args: Any, **kwargs: Any) -> Any:
        """Register a dynamic system prompt.

        System prompts can access runtime through RunContext[AgentContext].

        Example:
            ```python
            @agent.system_prompt
            async def get_prompt(ctx: RunContext[AgentContext]) -> str:
                resources = await ctx.deps.list_resource_names()
                return f"Available resources: {', '.join(resources)}"
            ```
        """
        return self._pydantic_agent.system_prompt(*args, **kwargs)

    # async def _handle_message(self, message: ChatMessage[Any]):
    #     """Handle a message from another agent. Can be used as signal slot."""
    #     msg = "_handle_message called on %s from %s with message %s"
    #     logger.debug(msg, self.name, source.name, message.content)
    #     # Convert any message to string for now as input
    #     await self.run(str(message.content))

    def clear_history(self):
        """Clear both internal and pydantic-ai history."""
        self._logger.clear_state()
        self._last_messages.clear()
        for tool in self._pydantic_agent._function_tools.values():
            tool.current_retry = 0
        logger.debug("Cleared history and reset tool state")

    def _handle_message(self, source: LLMlingAgent[Any, Any], message: ChatMessage[Any]):
        """Handle a message forwarded from another agent."""
        msg = "_handle_message called on %s from %s with message %s"
        logger.debug(msg, self.name, source.name, message.content)
        # await self.run(str(message.content), deps=source)
        loop = asyncio.get_event_loop()
        task = loop.create_task(self.run(str(message.content), deps=source))  # type: ignore[arg-type]
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    def set_model(self, model: models.Model | models.KnownModelName | None):
        """Set the model for this agent.

        Args:
            model: New model to use (name or instance)

        Emits:
            model_changed signal with the new model
        """
        old_name = self.model_name
        if isinstance(model, str):
            model = infer_model(model)
        self._pydantic_agent.model = model
        self.model_changed.emit(model)
        logger.debug("Changed model from %s to %s", old_name, self.model_name)

    def result_validator(self, *args: Any, **kwargs: Any) -> Any:
        """Register a result validator.

        Validators can access runtime through RunContext[AgentContext].

        Example:
            ```python
            @agent.result_validator
            async def validate(ctx: RunContext[AgentContext], result: str) -> str:
                if len(result) < 10:
                    raise ModelRetry("Response too short")
                return result
            ```
        """
        return self._pydantic_agent.result_validator(*args, **kwargs)

    @property
    def last_run_messages(self) -> list[ChatMessage]:
        """Get messages from the last run converted to our format."""
        return [convert_model_message(msg) for msg in self._last_messages]

    @property
    def runtime(self) -> RuntimeConfig:
        """Get the runtime configuration."""
        return self._runtime

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name

    @property
    def tools(self) -> ToolManager:
        return self._tool_manager


if __name__ == "__main__":
    import logging

    from llmling_agent import config_resources

    logging.basicConfig(level=logging.INFO)

    sys_prompt = "Open browser with google, please"

    async def main():
        async with RuntimeConfig.open(config_resources.OPEN_BROWSER) as r:
            agent: LLMlingAgent = LLMlingAgent(r, model="openai:gpt-4o-mini")
            result = await agent.run(sys_prompt)
            print(result.data)

    asyncio.run(main())
