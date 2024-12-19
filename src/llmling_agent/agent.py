"""LLMling integration with PydanticAI for AI-powered resource interaction."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence  # noqa: TC003
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from llmling.config.runtime import RuntimeConfig
from pydantic_ai import Agent as PydanticAgent, messages
from sqlmodel import Session
from typing_extensions import TypeVar

from llmling_agent.context import AgentContext
from llmling_agent.log import get_logger
from llmling_agent.models import AgentsManifest, TokenAndCostResult
from llmling_agent.pydantic_ai_utils import extract_token_usage_and_cost
from llmling_agent.responses import resolve_response_type
from llmling_agent.responses.models import InlineResponseDefinition
from llmling_agent.storage import Conversation, engine
from llmling_agent.storage.models import Message
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
    ) -> None:
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
        self._context = context or AgentContext.create_default(name)
        self._context.runtime = runtime

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

        self._enable_logging = enable_logging
        self._conversation_id = str(uuid4())

        if enable_logging:
            # Log conversation start
            with Session(engine) as session:
                id_ = self._conversation_id
                convo = Conversation(id=id_, agent_name=name)
                session.add(convo)
                session.commit()

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
        end_strategy: EndStrategy = "early",
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

    @property
    def model_name(self) -> str | None:
        """Get the model name in a consistent format."""
        match self._pydantic_agent.model:
            case str() | None:
                return self._pydantic_agent.model
            case _:
                return self._pydantic_agent.model.name()

    def _log_message(
        self,
        content: str,
        role: Literal["user", "assistant", "system"],
        *,
        cost_info: TokenAndCostResult | None = None,
        model: str | None = None,
    ) -> None:
        """Log a single message to the database.

        Args:
            content: Message content
            role: Message role (user/assistant/system)
            cost_info: Combined token usage and cost information
            model: Optional model name used
        """
        if not self._enable_logging:
            return

        with Session(engine) as session:
            msg = Message(
                conversation_id=self._conversation_id,
                role=role,
                content=content,
                token_usage=cost_info.token_usage if cost_info else None,
                cost=cost_info.cost_usd if cost_info else None,
                model=model,
            )
            session.add(msg)
            session.commit()

    async def run(
        self,
        prompt: str,
        *,
        message_history: list[ModelMessage] | None = None,
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
            # Clear all tools
            self._pydantic_agent._function_tools.clear()

            # Register currently enabled tools
            enabled_tools = self.tools.get_tools(state="enabled")
            for tool in enabled_tools:
                assert tool._original_callable
                self._pydantic_agent.tool_plain(tool._original_callable)

            logger.debug("agent run prompt=%s", prompt)
            logger.debug("  preparing model and tools run_step=%d", 1)

            # Run through pydantic-ai's public interface
            result = await self._pydantic_agent.run(
                prompt,
                deps=self._context,
                message_history=message_history,
                model=model,
            )

            if self._enable_logging:
                # Log user message
                self._log_message(prompt, role="user")

                # Get cost info for assistant response
                result_str = str(result.data)
                model_name = self.model_name
                cost = (
                    await extract_token_usage_and_cost(
                        result.usage(), model_name, prompt, result_str
                    )
                    if model_name
                    else None
                )

                # Log assistant response with all info
                self._log_message(
                    result_str,
                    role="assistant",
                    cost_info=cost,
                    model=model_name,
                )
        except Exception:
            logger.exception("Agent run failed")
            raise
        else:
            return result

    async def run_stream(
        self,
        prompt: str,
        *,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentContext, TResult]]:
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
            # Update pydantic agent's tools
            self._pydantic_agent._function_tools.clear()
            enabled_tools = self.tools.get_tools(state="enabled")
            for tool in enabled_tools:
                assert tool._original_callable
                self._pydantic_agent.tool_plain(tool._original_callable)

            if self._enable_logging:
                self._log_message(prompt, role="user")

            # Return the context manager directly - no await needed
            return self._pydantic_agent.run_stream(
                prompt,
                message_history=message_history,
                model=model,
                deps=self._context,
            )

        except Exception:
            logger.exception("Agent stream failed")
            raise

    def run_sync(
        self,
        prompt: str,
        *,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> RunResult[TResult]:
        """Run agent synchronously (convenience wrapper).

        Args:
            prompt: User query or instruction
            message_history: Optional previous messages for context
            model: Optional model override

        Returns:
            Result containing response and run information
        """
        try:
            return asyncio.run(
                self.run(prompt, message_history=message_history, model=model)
            )
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.exception("Sync agent run failed")
            raise

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
    def last_run_messages(self) -> list[messages.ModelMessage] | None:
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

    @property
    def tools(self) -> ToolManager:
        return self._tool_manager


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

    asyncio.run(main())
