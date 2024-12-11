"""LLMling integration with PydanticAI for AI-powered resource interaction."""

from __future__ import annotations

import asyncio
from collections.abc import (
    Sequence,  # noqa: TC003
)
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
import inspect
from inspect import Parameter, Signature
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from llmling.config.runtime import RuntimeConfig
from pydantic_ai import Agent as PydanticAgent, RunContext, messages
from pydantic_ai.result import RunResult, StreamedRunResult
from sqlmodel import Session
from typing_extensions import TypeVar

from llmling_agent.log import get_logger
from llmling_agent.storage import Conversation, engine
from llmling_agent.storage.models import Message


if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterator,
        Awaitable,
        Callable,
    )
    import os

    from llmling.core.events import Event
    from py2openai import OpenAIFunctionTool
    from pydantic_ai.agent import models
    from pydantic_ai.tools import ToolFuncPlain, ToolParams


logger = get_logger(__name__)

TResult = TypeVar("TResult", default=str)
T = TypeVar("T")  # For the return type

POS_OR_KEY = Parameter.POSITIONAL_OR_KEYWORD

JINJA_PROC = "jinja_template"  # Name of builtin LLMling Jinja2 processor


@dataclass
class ResourceInfo:
    """Information about an available resource.

    This class provides essential information about a resource that can be loaded.
    Use the resource name with load_resource() to access the actual content.
    """

    name: str
    """Name of the resource, use this with load_resource()"""

    uri: str
    """URI identifying the resource location"""

    description: str | None = None
    """Optional description of the resource's content or purpose"""


def _create_tool_wrapper(
    name: str,
    schema: OpenAIFunctionTool,
    original_callable: Callable[..., T | Awaitable[T]] | None = None,
) -> Callable[..., Awaitable[T]]:
    """Create a tool wrapper function with proper signature and type hints.

    Creates an async wrapper function that forwards calls to RuntimeConfig.execute_tool.
    If the original callable is provided, its signature and type hints are preserved.
    Otherwise, the signature is reconstructed from the OpenAI function schema.

    Args:
        name: Name of the tool to wrap
        schema: OpenAI function schema (from py2openai)
        original_callable: Optional original function to preserve signature from

    Returns:
        Async wrapper function with proper signature that delegates to execute_tool
    """
    # If we have the original callable, use its signature
    if original_callable:
        # Create parameters with original types
        sig = inspect.signature(original_callable)
        params = [
            Parameter("ctx", POS_OR_KEY, annotation=RunContext[RuntimeConfig]),
            *[
                Parameter(name, p.kind, annotation=p.annotation, default=p.default)
                for name, p in sig.parameters.items()
            ],
        ]
        return_annotation = sig.return_annotation
    else:
        # Fall back to schema-based parameters with Any types
        params = [Parameter("ctx", POS_OR_KEY, annotation=RunContext[RuntimeConfig])]
        properties = schema["function"].get("parameters", {}).get("properties", {})
        for prop_name, info in properties.items():
            default = Parameter.empty if info.get("required") else None
            param = Parameter(prop_name, POS_OR_KEY, annotation=Any, default=default)
            params.append(param)
        return_annotation = Any

    # Create the signature
    sig = Signature(params, return_annotation=return_annotation)

    # Create the wrapper function
    async def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = args[0]  # First arg is always context
        return await ctx.deps.execute_tool(name, **kwargs)

    # Apply the signature and metadata
    tool_wrapper.__signature__ = sig  # type: ignore
    tool_wrapper.__name__ = schema["function"]["name"]
    tool_wrapper.__doc__ = schema["function"]["description"]
    tool_wrapper.__annotations__ = {p.name: p.annotation for p in params}

    return tool_wrapper


class LLMlingAgent[TResult]:
    """Agent for AI-powered interaction with LLMling resources and tools.

    This agent integrates LLMling's resource system with PydanticAI's agent capabilities.
    It provides:
    - Access to resources through RuntimeConfig
    - Structured output support
    - Tool registration for resource operations
    - System prompt customization
    - Message history management

    Example:
        ```python
        # Simple text agent
        agent = LLMlingAgent(runtime)
        result = await agent.run("Load and summarize test.txt")
        print(result.data)  # Text summary

        # Agent with structured output
        class Analysis(BaseModel):
            summary: str
            complexity: int

        agent = LLMlingAgent[Analysis](
            runtime,
            result_type=Analysis,
        )
        result = await agent.run("Analyze test.txt")
        print(result.data.summary)  # Structured analysis
        ```
    """

    def __init__(
        self,
        runtime: RuntimeConfig,
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
        enable_logging: bool = True,
        **kwargs,
    ) -> None:
        """Initialize agent with runtime configuration.

        Args:
            runtime: Runtime configuration providing access to resources/tools
            result_type: Optional type for structured responses
            model: The default model to use (defaults to GPT-4)
            system_prompt: Static system prompts to use for this agent
            name: Name of the agent for logging
            retries: Default number of retries for failed operations
            result_tool_name: Name of the tool used for final result
            result_tool_description: Description of the final result tool
            result_retries: Max retries for result validation (defaults to retries)
            defer_model_check: Whether to defer model evaluation until first run
            kwargs: Additional arguments for PydanticAI agent
            enable_logging: Whether to enable logging for the agent
        """
        self._runtime = runtime

        # Use provided type or default to str
        actual_result_type = result_type or str

        # Initialize base PydanticAI agent
        self._pydantic_agent = PydanticAgent(
            model=model,
            result_type=actual_result_type,
            system_prompt=system_prompt,
            deps_type=RuntimeConfig,
            retries=retries,
            result_tool_name=result_tool_name,
            result_tool_description=result_tool_description,
            result_retries=result_retries,
            defer_model_check=defer_model_check,
            **kwargs,
        )
        # Set up event handling
        self._runtime.add_event_handler(self)
        self._setup_runtime_tools()
        self._name = name
        msg = "Initialized %s (model=%s, result_type=%s)"
        logger.debug(msg, self._name, model, result_type or "str")
        self._enable_logging = enable_logging
        self._conversation_id = str(uuid4())

        if enable_logging:
            # Log conversation start
            with Session(engine) as session:
                session.add(
                    Conversation(
                        id=self._conversation_id,
                        agent_name=name,
                        start_time=datetime.now(),
                    )
                )
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
            async with LLMlingAgent.open(
                "config.yml",
                model="openai:gpt-3.5-turbo"
            ) as agent:
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

    def _setup_runtime_tools(self) -> None:
        """Register all tools from runtime configuration."""
        for name, llm_tool in self.runtime.tools.items():
            schema = llm_tool.get_schema()
            wrapper: Callable[..., Any] = _create_tool_wrapper(name, schema)
            self.tool(wrapper)
            msg = "Registered runtime tool: %s (signature: %s)"
            logger.debug(msg, name, wrapper.__signature__)  # type: ignore

    async def handle_event(self, event: Event) -> None:
        """Handle runtime events.

        Override this method to add custom event handling.
        """
        # Default implementation just logs
        logger.debug("Received event: %s", event)

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
            result = await self._pydantic_agent.run(
                prompt,
                deps=self._runtime,
                message_history=message_history,
                model=model,
            )

            if self._enable_logging:
                now = datetime.now()
                cost = result.cost()

                token_usage: dict[str, int] | None = None
                if (
                    cost
                    and cost.total_tokens is not None
                    and cost.request_tokens is not None
                    and cost.response_tokens is not None
                ):
                    token_usage = {
                        "total": cost.total_tokens,
                        "prompt": cost.request_tokens,
                        "completion": cost.response_tokens,
                    }

                with Session(engine) as session:
                    # Log user message
                    session.add(
                        Message(
                            conversation_id=self._conversation_id,
                            timestamp=now,
                            role="user",
                            content=prompt,
                        )
                    )
                    # Log assistant response
                    session.add(
                        Message(
                            conversation_id=self._conversation_id,
                            timestamp=now,
                            role="assistant",
                            content=str(result.data),
                            token_usage=token_usage,
                            model=str(model or self._pydantic_agent.model),
                        )
                    )
                    session.commit()

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
                now = datetime.now()
                with Session(engine) as session:
                    session.add(
                        Message(
                            conversation_id=self._conversation_id,
                            timestamp=now,
                            role="user",
                            content=prompt,
                        )
                    )
                    session.commit()

            result = self._pydantic_agent.run_stream(
                prompt,
                deps=self._runtime,
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

    def tool(self, *args: Any, **kwargs: Any) -> Any:
        """Register a tool with the agent.

        Tools can access runtime through RunContext[RuntimeConfig].

        Example:
            ```python
            @agent.tool
            async def my_tool(ctx: RunContext[RuntimeConfig], arg: str) -> str:
                resource = await ctx.deps.load_resource(arg)
                return resource.content
            ```
        """
        return self._pydantic_agent.tool(*args, **kwargs)

    def tool_plain(self, func: ToolFuncPlain[ToolParams]) -> Any:
        """Register a plain tool with the agent.

        Plain tools don't receive runtime context.

        Example:
            ```python
            @agent.tool_plain
            def my_tool(arg: str) -> str:
                return arg.upper()
            ```
        """
        return self._pydantic_agent.tool_plain(func)

    def system_prompt(self, *args: Any, **kwargs: Any) -> Any:
        """Register a dynamic system prompt.

        System prompts can access runtime through RunContext[RuntimeConfig].

        Example:
            ```python
            @agent.system_prompt
            async def get_prompt(ctx: RunContext[RuntimeConfig]) -> str:
                resources = await ctx.deps.list_resource_names()
                return f"Available resources: {', '.join(resources)}"
            ```
        """
        return self._pydantic_agent.system_prompt(*args, **kwargs)

    def result_validator(self, *args: Any, **kwargs: Any) -> Any:
        """Register a result validator.

        Validators can access runtime through RunContext[RuntimeConfig].

        Example:
            ```python
            @agent.result_validator
            async def validate(ctx: RunContext[RuntimeConfig], result: str) -> str:
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


if __name__ == "__main__":
    import logging

    from llmling_agent import config_resources

    logging.basicConfig(level=logging.INFO)

    sys_prompt = "Open browser with google, please"

    # async def main() -> None:
    #     async with RuntimeConfig.open(config_resources.OPEN_BROWSER) as r:
    #         agent: LLMlingAgent[str] = LLMlingAgent(r, model="openai:gpt-3.5-turbo")
    #         result = await agent.run(sys_prompt)
    #         print(result.data)

    sys_prompt = "Check your resources and summarize the readme"

    async def main() -> None:
        async with RuntimeConfig.open(config_resources.SUMMARIZE_README) as r:
            agent: LLMlingAgent[str] = LLMlingAgent(r, model="openai:gpt-3.5-turbo")
            result = await agent.run(sys_prompt)
            print(result.data)

    asyncio.run(main())
