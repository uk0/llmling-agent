"""LLMling integration with PydanticAI for AI-powered resource interaction."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence  # noqa: TC003
from contextlib import asynccontextmanager
import time
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID, uuid4

from llmling import Config
from llmling.config.runtime import RuntimeConfig
from llmling.tools import LLMCallableTool, ToolError
from psygnal import Signal
from pydantic_ai import Agent as PydanticAgent, RunContext
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import infer_model
from typing_extensions import TypeVar

from llmling_agent.agent.conversation import ConversationManager
from llmling_agent.log import get_logger
from llmling_agent.models import AgentContext, AgentsManifest
from llmling_agent.models.agents import ToolCallInfo
from llmling_agent.models.messages import ChatMessage
from llmling_agent.pydantic_ai_utils import extract_usage, get_tool_calls, register_tool
from llmling_agent.responses import InlineResponseDefinition, resolve_response_type
from llmling_agent.tools.manager import ToolManager


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    import os

    from pydantic_ai.agent import EndStrategy, models
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.result import RunResult, StreamedRunResult

    from llmling_agent.tools.base import ToolInfo


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
        session_id: str | UUID | None = None,
        model: models.Model | models.KnownModelName | None = None,
        system_prompt: str | Sequence[str] = (),
        name: str = "llmling-agent",
        description: str | None = None,
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
            session_id: Optional id to recover a conversation
            model: The default model to use (defaults to GPT-4)
            system_prompt: Static system prompts to use for this agent
            name: Name of the agent for logging
            description: Description of the Agent ("what it can do")
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
        logger.debug("Runtime tools: %s", list(runtime.tools.keys()))
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
        self.name = name
        self.description = description
        msg = "Initialized %s (model=%s, result_type=%s)"
        logger.debug(msg, self.name, model, result_type or "str")

        from llmling_agent.agent import AgentLogger
        from llmling_agent.events import EventManager

        self._logger = AgentLogger(self, enable_logging=enable_logging)
        self._events = EventManager(self, enable_events=True)
        config_prompts = context.config.system_prompts if context else []
        all_prompts = list(config_prompts)
        if isinstance(system_prompt, str):
            all_prompts.append(system_prompt)
        else:
            all_prompts.extend(system_prompt)

        # Initialize ConversationManager with all prompts
        self.conversation = ConversationManager(
            self,
            initial_prompts=all_prompts,
            session_id=session_id,
        )

        self._pending_tasks: set[asyncio.Task[Any]] = set()
        self._connected_agents: set[LLMlingAgent[Any, Any]] = set()

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._pydantic_agent.name or "llmling-agent"

    @name.setter
    def name(self, value: str | None):
        self._pydantic_agent.name = value

    @classmethod
    @asynccontextmanager
    async def open(
        cls,
        config_path: str | os.PathLike[str] | Config | None = None,
        result_type: type[TResult] | None = None,
        *,
        model: models.Model | models.KnownModelName | None = None,
        session_id: str | UUID | None = None,
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
            config_path: Path to the runtime configuration file or a Config instance
                         (defaults to Config())
            result_type: Optional type for structured responses
            model: The default model to use (defaults to GPT-4)
            session_id: Optional id to recover a conversation
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
        if config_path is None:
            config_path = Config()
        async with RuntimeConfig.open(config_path) as runtime:
            agent = cls(
                runtime=runtime,
                result_type=result_type,
                model=model,
                session_id=session_id,
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
        session_id: str | UUID | None = None,
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

            # Basic Configuration
            model: Optional model override
            result_type: Optional type for structured responses
            model_settings: Additional model-specific settings
            session_id: Optional id to recover a conversation

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
        context = AgentContext[TDeps](
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
                session_id=session_id,
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
        logger.debug(
            "forwarding message from %s: %s (type: %s) to %d connected agents",
            self.name,
            repr(message.content),
            type(message.content),
            len(self._connected_agents),
        )
        self.outbox.emit(self, message)

    async def disconnect_all(self):
        """Disconnect from all agents."""
        if self._connected_agents:
            for target in list(self._connected_agents):
                self.stop_passing_results_to(target)

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
            register_tool(self._pydantic_agent, tool)

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
            message_id = str(uuid4())

            # Run through pydantic-ai's public interface
            start_time = time.perf_counter()
            msg_history = (
                message_history if message_history else self.conversation.get_history()
            )
            result = await self._pydantic_agent.run(
                prompt,
                deps=self._context,
                message_history=msg_history,
                model=model,
            )
            messages = result.new_messages()
            for call in get_tool_calls(messages):
                call.message_id = message_id
                call.context_data = self._context.data if self._context else None
                self.tool_used.emit(call)
            self.conversation._last_messages = list(messages)
            if not message_history:
                self.conversation.set_history(result.all_messages())
            # Emit user messages
            user_msg = ChatMessage[str](content=prompt, role="user")
            self.message_received.emit(user_msg)
            logger.debug("Agent run result: %r", result.data)
            # Get cost info for assistant response
            result_str = str(result.data)
            usage = result.usage()
            cost_info = (
                await extract_usage(usage, self.model_name, prompt, result_str)
                if self.model_name
                else None
            )

            # Create and emit assistant message
            assistant_msg = ChatMessage[TResult](
                content=result.data,
                role="assistant",
                name=self.name,
                model=self.model_name,
                message_id=message_id,
                cost_info=cost_info,
                response_time=time.perf_counter() - start_time,
            )
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

    def to_agent_tool(
        self,
        *,
        name: str | None = None,
        reset_history_on_run: bool = True,
        pass_message_history: bool = False,
        share_context: bool = False,
        parent: LLMlingAgent[Any, Any] | None = None,
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

        async def wrapped_tool(ctx: RunContext[AgentContext], prompt: str) -> str:
            if pass_message_history and not parent:
                msg = "Parent agent required for message history sharing"
                raise ToolError(msg)

            if reset_history_on_run:
                self.conversation.clear()

            history = (
                parent.conversation.get_history()
                if pass_message_history and parent
                else None
            )
            deps = ctx.deps.data if share_context else None

            result = await self.run(prompt, message_history=history, deps=deps)
            return str(result.data)

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
        prompt: str,
        *,
        deps: TDeps | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> AsyncIterator[StreamedRunResult[AgentContext[TDeps], TResult]]:
        """Run agent with prompt and get streaming response."""
        if deps is not None:
            self._context.data = deps
        try:
            self._update_tools()

            # Emit user message
            user_msg = ChatMessage[str](content=prompt, role="user")
            self.message_received.emit(user_msg)
            start_time = time.perf_counter()
            msg_history = (
                message_history if message_history else self.conversation.get_history()
            )

            # Capture all messages from the entire operation
            async with self._pydantic_agent.run_stream(
                prompt,
                message_history=msg_history,
                model=model,
                deps=self._context,
            ) as stream:
                original_stream = stream.stream

                async def wrapped_stream(*args, **kwargs):
                    async for chunk in original_stream(*args, **kwargs):
                        self.chunk_streamed.emit(str(chunk))
                        yield chunk

                    if stream.is_complete:
                        message_id = str(uuid4())
                        if not message_history:
                            self.conversation.set_history(stream.all_messages())
                        # Get complete result from the final chunks
                        if stream.is_structured:
                            message = stream._stream_response.get(final=True)
                            if not isinstance(message, ModelResponse):
                                msg = "Expected ModelResponse for structured output"
                                raise TypeError(msg)  # noqa: TRY301
                            result = await stream.validate_structured_result(message)
                        else:
                            # For text response, get() returns list[str]
                            chunks: list[str] = stream._stream_response.get(final=True)  # type: ignore
                            text = "".join(chunks)
                            result = cast(
                                TResult, await stream._validate_text_result(text)
                            )

                        usage = stream.usage()
                        cost_info = (
                            await extract_usage(
                                usage, self.model_name, prompt, str(result)
                            )
                            if self.model_name
                            else None
                        )

                        # Handle captured tool calls
                        messages = stream.new_messages()
                        self.conversation._last_messages = list(messages)
                        for call in get_tool_calls(messages):
                            call.message_id = message_id
                            call.context_data = (
                                self._context.data if self._context else None
                            )
                            self.tool_used.emit(call)

                        # Create and emit assistant message
                        assistant_msg = ChatMessage[TResult](
                            content=cast(TResult, result),
                            role="assistant",
                            name=self.name,
                            model=self.model_name,
                            message_id=message_id,
                            cost_info=cost_info,
                            response_time=time.perf_counter() - start_time,
                        )
                        self.message_sent.emit(assistant_msg)

                stream.stream = wrapped_stream  # type: ignore
                yield stream

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

    def clear_history(self):
        """Clear both internal and pydantic-ai history."""
        self._logger.clear_state()
        self.conversation.clear()
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

    def register_worker(
        self,
        worker: LLMlingAgent[Any, Any],
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
    def runtime(self) -> RuntimeConfig:
        """Get the runtime configuration."""
        return self._runtime

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
            agent = LLMlingAgent[Any, Any](r, model="openai:gpt-4o-mini")
            result = await agent.run(sys_prompt)
            print(result.data)

    asyncio.run(main())
