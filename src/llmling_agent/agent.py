"""LLMling integration with PydanticAI for AI-powered resource interaction."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence  # noqa: TC003
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from llmling.config.runtime import RuntimeConfig
from pydantic_ai import Agent as PydanticAgent, RunContext, Tool, messages
from pydantic_ai.result import RunResult, StreamedRunResult
from sqlmodel import Session
from typing_extensions import TypeVar

from llmling_agent.context import AgentContext
from llmling_agent.log import get_logger
from llmling_agent.models import AgentsManifest, resolve_response_type
from llmling_agent.pydantic_ai_utils import TokenUsage, extract_token_usage
from llmling_agent.storage import Conversation, engine
from llmling_agent.storage.models import Message
from llmling_agent.tools import ToolConfirmation, ToolContext
from llmling_agent.tools.base import create_runtime_tool_wrapper


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    import os

    from llmling.core.events import Event
    from py2openai import OpenAIFunctionTool
    from pydantic_ai.agent import models
    from pydantic_ai.tools import ToolFuncEither, ToolFuncPlain, ToolParams


logger = get_logger(__name__)

TResult = TypeVar("TResult", default=str)
T = TypeVar("T")  # For the return type


JINJA_PROC = "jinja_template"  # Name of builtin LLMling Jinja2 processor


class LLMlingAgent[TResult]:
    """Agent for AI-powered interaction with LLMling resources and tools.

    This agent integrates LLMling's resource system with PydanticAI's agent capabilities.
    It provides:
    - Access to resources through RuntimeConfig
    - Structured output support
    - Tool registration for resource operations
    - System prompt customization
    - Message history management
    """

    def __init__(
        self,
        runtime: RuntimeConfig,
        context: AgentContext | None = None,
        result_type: type[TResult] | None = None,
        *,
        model: models.Model | models.KnownModelName | None = None,
        system_prompt: str | Sequence[str] = (),
        name: str = "llmling-agent",
        tool_confirmation: ToolConfirmation | None = None,
        confirm_tools: set[str] | bool = False,
        tools: Sequence[Tool[AgentContext]] = (),
        retries: int = 1,
        result_tool_name: str = "final_result",
        result_tool_description: str | None = None,
        result_retries: int | None = None,
        tool_choice: bool | str | list[str] = True,
        defer_model_check: bool = False,
        enable_logging: bool = True,
        **kwargs,
    ) -> None:
        """Initialize agent with runtime configuration.

        Args:
            runtime: Runtime configuration providing access to resources/tools
            context: Agent context with capabilities and configuration
            result_type: Optional type for structured responses
            model: The default model to use (defaults to GPT-4)
            system_prompt: Static system prompts to use for this agent
            name: Name of the agent for logging
            tool_confirmation: Callback class for tool confirmation
            confirm_tools: List of tools requiring confirmation or global on / off
            tools: List of tools to register with the agent
            retries: Default number of retries for failed operations
            result_tool_name: Name of the tool used for final result
            result_tool_description: Description of the final result tool
            result_retries: Max retries for result validation (defaults to retries)
            tool_choice: Ability to set a fixed tool or temporarily disable tools usage.
            defer_model_check: Whether to defer model evaluation until first run
            kwargs: Additional arguments for PydanticAI agent
            enable_logging: Whether to enable logging for the agent
        """
        self._runtime = runtime
        self._context = context or AgentContext.create_default(name)
        self._context.runtime = runtime
        self._tools = list(tools)
        self._disabled_tools: set[str] = set()
        self._original_tools = list(tools)  # Store original tools for reference
        # Tool confirmation setup
        self._tool_confirmation = tool_confirmation
        self._confirm_tools = self._setup_tool_confirmation(confirm_tools)
        self._tool_choice = tool_choice

        # Prepare all tools including runtime tools
        tools = list(self._tools)  # Start with registered tools
        tools = self._prepare_tools()
        self._setup_history_tools(tools)

        # Resolve result type
        actual_type: type[TResult]
        if isinstance(result_type, str):
            actual_type = resolve_response_type(result_type, context)  # type: ignore[assignment]
        else:
            actual_type = result_type or str  # type: ignore[assignment]

        # Initialize agent with all tools
        self._pydantic_agent = PydanticAgent(
            model=model,
            result_type=actual_type,
            system_prompt=system_prompt,
            deps_type=AgentContext,
            tools=tools,
            retries=retries,
            result_tool_name=result_tool_name,
            result_tool_description=result_tool_description,
            result_retries=result_retries,
            defer_model_check=defer_model_check,
            **kwargs,
        )
        # Set up event handling
        self._runtime.add_event_handler(self)

        self._name = name
        msg = "Initialized %s (model=%s, result_type=%s)"
        logger.debug(msg, self._name, model, result_type or "str")

        self._enable_logging = enable_logging
        self._conversation_id = str(uuid4())

        if enable_logging:
            # Log conversation start
            with Session(engine) as session:
                id_ = self._conversation_id
                convo = Conversation(id=id_, agent_name=name)
                session.add(convo)
                session.commit()

    def _setup_history_tools(self, tools: list[Tool[AgentContext]]) -> None:
        """Set up history-related tools based on capabilities."""
        if not self._context:
            return

        if self._context.capabilities.history_access != "none":
            from llmling_agent.tools.history import HistoryTools

            history_tools = HistoryTools(self._context)
            tools.append(Tool(history_tools.search_history, takes_ctx=True))
            if self._context.capabilities.stats_access != "none":
                tools.append(Tool(history_tools.show_statistics, takes_ctx=True))

    async def _confirm_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        schema: OpenAIFunctionTool,
        runtime_ctx: RunContext[AgentContext],
    ) -> bool:
        """Request confirmation for tool execution.

        Args:
            tool_name: Name of the tool to execute
            args: Tool arguments
            schema: Tool's OpenAI function schema
            runtime_ctx: Current runtime context

        Returns:
            Whether execution was confirmed
        """
        if (
            not self._tool_confirmation
            or self._confirm_tools is None
            or tool_name not in self._confirm_tools
        ):
            return True

        tool_ctx = ToolContext(
            name=tool_name,
            args=args,
            schema=schema,
            runtime_ctx=runtime_ctx,
        )
        return await self._tool_confirmation.confirm_tool(tool_ctx)

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
        defer_model_check: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[LLMlingAgent[TResult]]:
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
        config: str | AgentsManifest,
        agent_name: str,
        *,
        # Model configuration
        model: str | models.Model | models.KnownModelName | None = None,
        result_type: type[TResult] | None = None,
        model_settings: dict[str, Any] | None = None,
        # Tool configuration
        tools: list[str | Callable[..., Any]] | None = None,
        tool_choice: bool | str | list[str] = True,
        confirm_tools: set[str] | bool = False,
        tool_confirmation: ToolConfirmation | None = None,
        # Execution settings
        retries: int = 1,
        result_tool_name: str = "final_result",
        result_tool_description: str | None = None,
        result_retries: int | None = None,
        # Other settings
        system_prompt: str | Sequence[str] | None = None,
        enable_logging: bool = True,
    ) -> AsyncIterator[LLMlingAgent[TResult]]:
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
            confirm_tools: Which tools need confirmation:
                - True: All tools
                - False: No tools
                - set[str]: Specific tools
            tool_confirmation: Optional callback for tool confirmation

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
                confirm_tools={"dangerous_tool"},
            ) as agent:
                result = await agent.run("Do something")
            ```
        """
        if isinstance(config, AgentsManifest):
            agent_def = config
        else:
            agent_def = AgentsManifest.from_file(config)

        if agent_name not in agent_def.agents:
            msg = f"Agent '{agent_name}' not found in {config}"
            raise ValueError(msg)

        agent_config = agent_def.agents[agent_name]

        # Use model from override or agent config
        actual_model = model or agent_config.model
        if not actual_model:
            msg = "Model must be specified either in config or as override"
            raise ValueError(msg)

        capabilities = agent_def.get_capabilities(agent_config.role)

        # Create context
        context = AgentContext(
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
                tool_confirmation=tool_confirmation,
                confirm_tools=confirm_tools,
                retries=retries,
                result_tool_name=result_tool_name,
                result_tool_description=result_tool_description,
                result_retries=result_retries,
                tool_choice=tool_choice,
                system_prompt=system_prompt or [],
                enable_logging=enable_logging,
            )
            try:
                yield agent
            finally:
                # Any cleanup if needed
                pass

    def _setup_tool_confirmation(self, confirm_tools: set[str] | bool) -> set[str] | None:
        """Set up tool confirmation configuration.

        Args:
            confirm_tools: What tools need confirmation
                           (True=all, set=specific tools, False=none)

        Returns:
            Set of tool names requiring confirmation, or None if disabled
        """
        match confirm_tools:
            case True:
                return set(self.runtime.tools)
            case set() | list():
                return set(confirm_tools)
            case False:
                return None

    @property
    def model_name(self) -> str | None:
        """Get the model name in a consistent format."""
        match self._pydantic_agent.model:
            case str() | None:
                return self._pydantic_agent.model
            case _:
                return self._pydantic_agent.model.name()

    async def handle_event(self, event: Event) -> None:
        """Handle runtime events.

        Override this method to add custom event handling.
        """
        # Default implementation just logs
        logger.debug("Received event: %s", event)

    def _log_message(
        self,
        content: str,
        role: Literal["user", "assistant", "system"],
        *,
        token_usage: TokenUsage | None = None,
        model: str | None = None,
    ) -> None:
        """Log a single message to the database.

        Args:
            content: Message content
            role: Message role (user/assistant/system)
            token_usage: Optional token usage statistics
            model: Optional model name used
        """
        if not self._enable_logging:
            return

        with Session(engine) as session:
            msg = Message(
                conversation_id=self._conversation_id,
                role=role,
                content=content,
                token_usage=token_usage,
                model=model,
            )
            session.add(msg)
            session.commit()

    async def run(
        self,
        prompt: str,
        *,
        message_history: list[messages.Message] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> RunResult[TResult]:
        """Run agent with prompt and get response.

        Args:
            prompt: User query or instruction
            message_history: Optional previous messages for context
            model: Optional model override

        Returns:
            Result containing response and run information

        Raises:
            UnexpectedModelBehavior: If the model fails or behaves unexpectedly
        """
        try:
            logger.debug("agent run prompt=%s", prompt)
            logger.debug("  preparing model and tools run_step=%d", 1)

            # Log what the model sees
            logger.debug("Funtion tools: %s", self._pydantic_agent._function_tools)
            logger.debug("System prompts: %s", self._pydantic_agent._system_prompts)
            result = await self._pydantic_agent.run(
                prompt,
                deps=self._context,
                message_history=message_history,
                model=model,
            )
            if self._enable_logging:
                # Log user message
                self._log_message(prompt, role="user")

                # Log assistant response with stats
                token_usage = extract_token_usage(result.cost())
                self._log_message(
                    str(result.data),
                    role="assistant",
                    token_usage=token_usage,
                    model=self.model_name,
                )

            return cast(RunResult[TResult], result)
        except Exception:
            logger.exception("Agent run failed")
            raise

    async def run_stream(
        self,
        prompt: str,
        *,
        message_history: list[messages.Message] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[TResult, messages.Message]]:
        """Run agent with prompt and get streaming response.

        Args:
            prompt: User query or instruction
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
        try:
            if self._enable_logging:
                self._log_message(prompt, role="user")

            result = self._pydantic_agent.run_stream(
                prompt,
                deps=self._context,
                message_history=message_history,
                model=model,
            )
            return cast(
                AbstractAsyncContextManager[StreamedRunResult[TResult, messages.Message]],
                result,
            )
        except Exception:
            logger.exception("Agent stream failed")
            raise

    def run_sync(
        self,
        prompt: str,
        *,
        message_history: list[messages.Message] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> RunResult[TResult]:
        """Run agent synchronously (convenience wrapper).

        Args:
            prompt: User query or instruction
            message_history:too Optional previous messages for context
            model: Optional model override

        Returns:
            Result containing response and run information

        Raises:
            UnexpectedModelBehavior: If the model fails or behaves unexpectedly
        """
        try:
            main = self.run(prompt, message_history=message_history, model=model)
            return asyncio.run(main)
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.exception("Sync agent run failed")
            raise

    def tool(
        self,
        func: ToolFuncEither[AgentContext, ...] | None = None,
        *,
        max_retries: int | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Register a tool with the agent.

        Can be used as a decorator or called directly.

        Example:
            ```python
            @agent.tool
            async def my_tool(ctx: RunContext[AgentContext], arg: str) -> str:
                caps = ctx.deps.capabilities
                if not caps.some_permission:
                    raise PermissionError("Permission denied")
                return f"Processed {arg}"
            ```
        """
        if func is None:
            return lambda f: self.tool(
                f, max_retries=max_retries, name=name, description=description
            )

        # Create Tool instance and append to tools list
        tool_instance: Tool[AgentContext] = Tool(
            func,
            takes_ctx=True,
            max_retries=max_retries,
            name=name,
            description=description,
        )
        self._tools.append(tool_instance)  # Collect tools for initialization
        return func

    def tool_plain(
        self,
        func: ToolFuncPlain[ToolParams] | None = None,
        *,
        max_retries: int | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Register a plain tool (without context) with the agent."""
        if func is None:
            return lambda f: self.tool_plain(
                f,
                max_retries=max_retries,
                name=name,
                description=description,
            )

        tool_instance: Tool[AgentContext] = Tool(
            func,
            takes_ctx=False,
            max_retries=max_retries,
            name=name,
            description=description,
        )
        self._tools.append(tool_instance)
        return func

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
    def last_run_messages(self) -> list[messages.Message] | None:
        """Get messages from the last run."""
        return self._pydantic_agent.last_run_messages

    @property
    def runtime(self) -> RuntimeConfig:
        """Get the runtime configuration."""
        return self._runtime

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name

    def enable_tool(self, tool_name: str) -> None:
        """Enable a previously disabled tool.

        Args:
            tool_name: Name of the tool to enable

        Raises:
            ValueError: If tool doesn't exist
        """
        if tool_name not in self.runtime.tools and not any(
            t.name == tool_name for t in self._original_tools
        ):
            msg = f"Tool {tool_name!r} not found"
            raise ValueError(msg)
        self._disabled_tools.discard(tool_name)
        logger.debug("Enabled tool: %s", tool_name)

    def disable_tool(self, tool_name: str) -> None:
        """Disable a tool."""
        if tool_name not in self.runtime.tools and not any(
            t.name == tool_name for t in self._original_tools
        ):
            msg = f"Tool '{tool_name}' not found"
            raise ValueError(msg)
        self._disabled_tools.add(tool_name)
        logger.debug("Disabled tool: %s", tool_name)

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is currently enabled.

        Args:
            tool_name: Name of the tool to check

        Returns:
            Whether the tool is enabled
        """
        return tool_name not in self._disabled_tools

    def list_tools(self) -> dict[str, bool]:
        """Get a mapping of all tools and their enabled status.

        Returns:
            Dict mapping tool names to their enabled status
        """
        tools = {name: name not in self._disabled_tools for name in self.runtime.tools}
        # Add custom tools
        tools.update({
            t.name: t.name not in self._disabled_tools for t in self._original_tools
        })
        return tools

    def _prepare_tools(self) -> list[Tool[AgentContext]]:
        """Prepare all tools respecting enabled/disabled state."""
        match self._tool_choice:
            case False:  # no tools
                return []
            case str() as tool_name:  # specific tool
                return self._get_tools([tool_name])
            case list() as tool_names:  # list of specific tools
                return self._get_tools(tool_names)
            case True:  # auto - return all enabled tools
                tools = list(self._tools)
                for tool_name in self.runtime.tools:
                    if (
                        tool_name not in self._disabled_tools
                    ):  # This could be refactored later
                        tool_def = self.runtime.tools[tool_name]
                        wrapped = create_runtime_tool_wrapper(
                            name=tool_name,
                            schema=tool_def.get_schema(),
                            description=tool_def.description,
                        )
                        tools.append(wrapped)
                return tools
            case _:
                return []

    def _get_tools(self, tool_names: list[str]) -> list[Tool[AgentContext]]:
        """Get specified tools from both runtime and custom tools."""
        tools = []
        for name in tool_names:
            # Add runtime tool if it matches
            if name in self.runtime.tools:
                tool_def = self.runtime.tools[name]
                wrapped = create_runtime_tool_wrapper(
                    name=name,
                    schema=tool_def.get_schema(),
                    description=tool_def.description,
                )
                tools.append(wrapped)
            # Add custom tool if it matches
            tools.extend(t for t in self._tools if t.name == name)
        return tools


if __name__ == "__main__":
    import logging

    from llmling_agent import config_resources

    logging.basicConfig(level=logging.INFO)

    sys_prompt = "Open browser with google, please"

    async def main() -> None:
        async with RuntimeConfig.open(config_resources.OPEN_BROWSER) as r:
            agent: LLMlingAgent[str] = LLMlingAgent(r, model="openai:gpt-4o-mini")
            result = await agent.run(sys_prompt)
            print(result.data)

    # sys_prompt = "Check your resources and summarize the readme"

    # async def main() -> None:
    #     async with RuntimeConfig.open(config_resources.SUMMARIZE_README) as r:
    #         agent: LLMlingAgent[str] = LLMlingAgent(r, model="openai:gpt-4o-mini")
    #         result = await agent.run(sys_prompt)
    #         print(result.data)

    asyncio.run(main())
