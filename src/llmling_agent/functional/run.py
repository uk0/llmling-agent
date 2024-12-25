"""High-level pipeline functions for agent execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

from llmling import Config

from llmling_agent import LLMlingAgent
from llmling_agent.environment.models import FileEnvironment, InlineEnvironment
from llmling_agent.log import get_logger
from llmling_agent.models import AgentConfig, AgentsManifest, SystemPrompt
from llmling_agent.responses import InlineResponseDefinition, ResponseField


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.agent import models

    from llmling_agent.environment import AgentEnvironment

logger = get_logger(__name__)

T = TypeVar("T")

OutputFormat = Literal["text", "json", "yaml", "raw"]
ErrorHandling = Literal["raise", "return", "ignore"]


@runtime_checkable
class PromptLike(Protocol):
    """Protocol for objects that can be converted to prompt strings."""

    def __str__(self) -> str: ...


def ensure_str(prompt: str | PromptLike) -> str:
    """Convert prompt-like object to string."""
    return str(prompt)


@overload
async def run_agent_pipeline(
    agent_name: str,
    prompt: str | list[str] | SystemPrompt,
    config: str | AgentsManifest,
    *,
    model: str | None = None,
    output_format: Literal["raw"] = "raw",
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
    result_type: type[T] | None = None,
    stream: Literal[False] = False,
    retries: int | None = None,
    capabilities: dict[str, bool] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[str | Callable[..., Any]] | None = None,
    model_settings: dict[str, Any] | None = None,
) -> T: ...


@overload
async def run_agent_pipeline(
    agent_name: str,
    prompt: str | list[str] | SystemPrompt,
    config: str | AgentsManifest,
    *,
    model: str | None = None,
    output_format: Literal["text", "json", "yaml"],
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
    result_type: type[T] | None = None,
    stream: Literal[False] = False,
    retries: int | None = None,
    capabilities: dict[str, bool] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[str | Callable[..., Any]] | None = None,
    model_settings: dict[str, Any] | None = None,
) -> str: ...


@overload
async def run_agent_pipeline(
    agent_name: str,
    prompt: str | list[str] | SystemPrompt,
    config: str | AgentsManifest,
    *,
    stream: Literal[True],
    model: str | None = None,
    output_format: OutputFormat = "text",
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
    result_type: type[T] | None = None,
    retries: int | None = None,
    capabilities: dict[str, bool] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[str | Callable[..., Any]] | None = None,
    model_settings: dict[str, Any] | None = None,
) -> AsyncIterator[str]: ...


async def run_agent_pipeline(  # noqa: PLR0911
    agent_name: str,
    prompt: str | list[str] | SystemPrompt,
    config: str | AgentsManifest,
    *,
    model: str | None = None,
    output_format: OutputFormat = "text",
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
    result_type: type[T] | None = None,
    stream: bool = False,
    retries: int | None = None,
    capabilities: dict[str, bool] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[str | Callable[..., Any]] | None = None,
    model_settings: dict[str, Any] | None = None,
) -> T | str | AsyncIterator[str]:
    """Execute complete agent pipeline.

    This is a high-level function that:
    1. Loads agent configuration
    2. Initializes the agent with specified settings
    3. Executes the prompt(s)
    4. Formats and returns the result

    Args:
        agent_name: Name of the agent to run
        prompt: Single prompt, list of prompts, or system prompt
        config: Path to agent configuration file or AgentsManifest instance
        model: Optional model override
        output_format: Output format (text/json/yaml/raw)
        environment: Override environment configuration:
            - str: Path to environment file
            - Config: Direct runtime configuration
            - AgentEnvironment: Complete environment definition
            - None: Use agent's default environment
        error_handling: How to handle errors:
            - raise: Raise exceptions (default)
            - return: Return error message as string
            - ignore: Return None for errors
        result_type: Expected result type (for validation)
        stream: Whether to stream responses
        retries: Number of retries for failed operations
        capabilities: Override agent capabilities
        tool_choice: Control tool usage:
            - True: Allow all tools
            - False: No tools
            - str: Use specific tool
            - list[str]: Allow specific tools
        tools: list of callables or import paths which can be used as tools
        model_settings: Additional model-specific settings

    Returns:
        - If stream=False: Formatted response or raw result
        - If stream=True: AsyncIterator yielding response chunks

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If agent initialization fails
        TypeError: If result type doesn't match expected type
    """
    try:
        # Load configuration if needed
        agent_def = (
            config
            if isinstance(config, AgentsManifest)
            else AgentsManifest.from_file(config)
        )

        if agent_name not in agent_def.agents:
            msg = f"Agent '{agent_name}' not found in configuration"
            raise ValueError(msg)  # noqa: TRY301

        # Prepare agent configuration
        agent_config = agent_def.agents[agent_name]
        if environment is not None:
            match environment:
                case str():
                    # Path to environment file
                    update: dict[str, Any] = {"environment": environment}
                    agent_config = agent_config.model_copy(update=update)
                case Config():
                    # Direct runtime config
                    update = {"environment": InlineEnvironment(config=environment)}
                    agent_config = agent_config.model_copy(update=update)
                case FileEnvironment() | InlineEnvironment():
                    # Complete environment definition
                    update = {"environment": environment}
                    agent_config = agent_config.model_copy(update=update)
                case _:
                    msg = f"Invalid environment type: {type(environment)}"
                    raise TypeError(msg)  # noqa: TRY301

        # Update capabilities if provided
        if capabilities and agent_config.role in agent_def.roles:
            current = agent_def.roles[agent_config.role].model_dump()
            current.update(capabilities)
            role = agent_def.roles[agent_config.role]
            agent_def.roles[agent_config.role] = role.model_copy(update=current)

        # Create agent with all settings
        async with LLMlingAgent[Any, T].open_agent(
            agent_def,
            agent_name,
            model=model,  # type: ignore[arg-type]
            result_type=result_type,
            retries=retries or 1,
            tools=tools,
            tool_choice=tool_choice,
            model_settings=model_settings or {},
        ) as agent:
            # Handle different prompt types
            prompts = (
                [ensure_str(prompt)]
                if isinstance(prompt, str | PromptLike)
                else [ensure_str(p) for p in prompt]
            )
            if stream:
                # Streaming mode - yield messages
                async def stream_prompts() -> AsyncIterator[str]:
                    last_messages = None
                    for p in prompts:
                        async with agent.run_stream(
                            p, message_history=last_messages
                        ) as result:
                            async for message in result.stream():
                                yield str(message)
                            last_messages = result.new_messages()

                # Return the async iterator
                return stream_prompts()

            # Non-streaming mode - return final result
            last_result = None
            last_messages = None

            for p in prompts:
                result = await agent.run(p, message_history=last_messages)
                last_result = result
                last_messages = result.new_messages()

            if not last_result:
                msg = "No result produced"
                raise RuntimeError(msg)  # noqa: TRY301

            # Format output based on format
            match output_format:
                case "raw":
                    return last_result.data  # type: ignore[return-value]
                case "json":
                    if hasattr(last_result.data, "model_dump_json"):
                        return last_result.data.model_dump_json(indent=2)  # pyright: ignore
                    return str(last_result.data)
                case "yaml":
                    if hasattr(last_result.data, "model_dump_yaml"):
                        return last_result.data.model_dump_yaml()  # pyright: ignore
                    return str(last_result.data)
                case "text" | _:
                    return str(last_result.data)

    except Exception as e:
        match error_handling:
            case "raise":
                raise
            case "return":
                return f"Error: {e}"
            case "ignore":
                return "" if output_format != "raw" else None  # type: ignore[return-value]
            case _:
                msg = f"Invalid error_handling: {error_handling}"
                raise ValueError(msg) from e


def run_agent_pipeline_sync(
    agent_name: str,
    prompt: str | list[str] | SystemPrompt,
    config: str | AgentsManifest,
    **kwargs: Any,
) -> Any:
    """Synchronous version of run_agent_pipeline.

    This is a convenience wrapper that runs the async pipeline in a new event loop.
    See run_agent_pipeline for full documentation.

    Note: Streaming mode is not supported in the sync version.
    """
    if kwargs.get("stream"):
        msg = "Streaming not supported in synchronous version"
        raise ValueError(msg)

    fn = run_agent_pipeline(agent_name=agent_name, prompt=prompt, config=config, **kwargs)
    return asyncio.run(fn)


@overload
async def run_with_model(
    prompt: str | list[str] | SystemPrompt,
    model: str | models.Model | models.KnownModelName,
    *,
    result_type: None = None,
    system_prompt: str | list[str] | None = None,
    output_format: Literal["text", "json", "yaml"] = "text",
    stream: Literal[False] = False,
    model_settings: dict[str, Any] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[str | Callable[..., Any]] | None = None,
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
) -> str: ...


@overload
async def run_with_model(
    prompt: str | list[str] | SystemPrompt,
    model: str | models.Model | models.KnownModelName,
    *,
    result_type: type[T],
    system_prompt: str | list[str] | None = None,
    output_format: OutputFormat = "raw",  # Allow any OutputFormat
    stream: Literal[False] = False,
    model_settings: dict[str, Any] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[str | Callable[..., Any]] | None = None,
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
) -> T: ...


@overload
async def run_with_model(
    prompt: str | list[str] | SystemPrompt,
    model: str | models.Model | models.KnownModelName,
    *,
    stream: Literal[True],
    result_type: type[T] | None = None,
    system_prompt: str | list[str] | None = None,
    output_format: OutputFormat = "text",
    model_settings: dict[str, Any] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[str | Callable[..., Any]] | None = None,
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
) -> AsyncIterator[str]: ...


async def run_with_model(
    prompt: str | list[str] | SystemPrompt,
    model: str | models.Model | models.KnownModelName,
    *,
    result_type: type[T] | None = None,
    system_prompt: str | list[str] | None = None,
    output_format: OutputFormat = "text",
    stream: bool = False,
    model_settings: dict[str, Any] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[str | Callable[..., Any]] | None = None,
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
) -> T | str | AsyncIterator[str]:
    """Run a prompt with a specific model.

    Simple interface to execute prompts with LLMs without complex agent configuration.

    Args:
        prompt: Prompt(s) to send to the model
        model: Model to use (name or instance)
        result_type: Expected result type for validation
        system_prompt: Optional system prompt(s)
        output_format: Output format (text/json/yaml/raw)
        stream: Whether to stream responses
        model_settings: Model-specific settings
        tool_choice: Control tool usage:
            - True: Allow all tools
            - False: No tools
            - str: Use specific tool
            - list[str]: Allow specific tools
        tools: list of callables or import paths which can be used as tools
        environment: Optional environment configuration
        error_handling: How to handle errors (raise/return/ignore)

    Returns:
        - If stream=False: Formatted response or raw result
        - If stream=True: AsyncIterator yielding response chunks

    Examples:
        # Simple text completion
        >>> result = await run_with_model("Hello!", "gpt-4")

        # Structured output
        >>> result = await run_with_model(
        ...     "Analyze this",
        ...     "gpt-4",
        ...     result_type=AnalysisResult,
        ...     system_prompt="You are an expert analyzer",
        ...     output_format="json"
        ... )

        # Streaming with tools
        >>> async for chunk in await run_with_model(
        ...     "Help me with task",
        ...     "gpt-4",
        ...     stream=True,
        ...     environment="tools.yml"
        ... ):
        ...     print(chunk)
    """
    # Create minimal manifest with optional response type
    responses = {}
    if result_type is not None:
        fields = {"result": ResponseField(type="str", description="Result")}
        r = InlineResponseDefinition(description="Default result type", fields=fields)
        responses["DefaultResult"] = r

    match system_prompt:
        case str():
            sys_prompts = [system_prompt]
        case Sequence():
            sys_prompts = list(system_prompt)
        case _:
            sys_prompts = []

    agent_environment = (
        InlineEnvironment(config=environment)
        if isinstance(environment, Config)
        else environment
    )
    cfg = AgentConfig(
        name="default",
        model=model,  # type: ignore
        system_prompts=sys_prompts,
        result_type="DefaultResult" if result_type else None,
        environment=agent_environment,
    )
    manifest = AgentsManifest(responses=responses, agents={"default": cfg})

    if stream:
        return await run_agent_pipeline(
            "default",
            prompt,
            manifest,
            output_format=output_format,
            stream=True,
            model_settings=model_settings,
            tool_choice=tool_choice,
            tools=tools,
            error_handling=error_handling,
            result_type=result_type,
        )
    return await run_agent_pipeline(
        "default",
        prompt,
        manifest,
        output_format=output_format,
        stream=False,
        model_settings=model_settings,
        tool_choice=tool_choice,
        tools=tools,
        error_handling=error_handling,
        result_type=result_type,
    )


@overload
def run_with_model_sync(
    prompt: str | list[str] | SystemPrompt,
    model: str | models.Model | models.KnownModelName,
    *,
    result_type: None = None,
    system_prompt: str | list[str] | None = None,
    output_format: Literal["text", "json", "yaml"] = "text",
    model_settings: dict[str, Any] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[str | Callable[..., Any]] | None = None,
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
) -> str: ...


@overload
def run_with_model_sync(
    prompt: str | list[str] | SystemPrompt,
    model: str | models.Model | models.KnownModelName,
    *,
    result_type: type[T],
    system_prompt: str | list[str] | None = None,
    output_format: Literal["raw"] = "raw",
    model_settings: dict[str, Any] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[str | Callable[..., Any]] | None = None,
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
) -> T: ...


def run_with_model_sync(
    prompt: str | list[str] | SystemPrompt,
    model: str | models.Model | models.KnownModelName,
    *,
    result_type: type[T] | None = None,
    system_prompt: str | list[str] | None = None,
    output_format: OutputFormat = "text",
    model_settings: dict[str, Any] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[str | Callable[..., Any]] | None = None,
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
) -> T | str:
    """Synchronous version of run_with_model.

    This is a convenience wrapper that runs the async pipeline in a new event loop.
    See run_with_model for full documentation.

    Note: Streaming mode is not supported in the sync version.

    Examples:
        # Simple usage
        >>> result = run_with_model_sync(
        ...     "Hello!",
        ...     "gpt-4",
        ...     system_prompt="Be concise"
        ... )

        # With structured output
        >>> result = run_with_model_sync(
        ...     "Analyze this",
        ...     "gpt-4",
        ...     result_type=AnalysisResult,
        ...     output_format="json"
        ... )
    """
    return asyncio.run(
        run_with_model(  # type: ignore
            prompt=prompt,
            model=model,
            result_type=result_type,
            system_prompt=system_prompt,
            output_format=output_format,
            stream=False,  # type: ignore[arg-type]
            model_settings=model_settings,
            tool_choice=tool_choice,
            tools=tools,
            environment=environment,
            error_handling=error_handling,
        )
    )
