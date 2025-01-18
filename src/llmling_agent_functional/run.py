"""High-level pipeline functions for agent execution."""

from __future__ import annotations

import asyncio
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
from toprompt import to_prompt

from llmling_agent import Agent
from llmling_agent.environment.models import FileEnvironment, InlineEnvironment
from llmling_agent.log import get_logger
from llmling_agent.models import AgentConfig, AgentsManifest
from llmling_agent.responses import InlineResponseDefinition, ResponseField


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from pydantic_ai.agent import models
    from toprompt import AnyPromptType

    from llmling_agent.common_types import ToolType
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
    prompt: AnyPromptType | list[AnyPromptType],
    config: str | AgentsManifest,
    *,
    model: str | None = None,
    output_format: Literal["raw"] = "raw",
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
    result_type: type[T] | None = None,
    retries: int | None = None,
    capabilities: dict[str, bool] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[ToolType] | None = None,
    model_settings: dict[str, Any] | None = None,
) -> T: ...


@overload
async def run_agent_pipeline(
    agent_name: str,
    prompt: AnyPromptType | list[AnyPromptType],
    config: str | AgentsManifest,
    *,
    model: str | None = None,
    output_format: OutputFormat,
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
    result_type: type[T] | None = None,
    retries: int | None = None,
    capabilities: dict[str, bool] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[ToolType] | None = None,
    model_settings: dict[str, Any] | None = None,
) -> str: ...


async def run_agent_pipeline(  # noqa: PLR0911
    agent_name: str,
    prompt: AnyPromptType | list[AnyPromptType],
    config: str | AgentsManifest,
    *,
    model: str | None = None,
    output_format: OutputFormat = "text",
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
    result_type: type[T] | None = None,
    retries: int | None = None,
    capabilities: dict[str, bool] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[ToolType] | None = None,
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
        retries: Number of retries for failed operations
        capabilities: Override agent capabilities
        tool_choice: Control tool usage:
            - True: Allow all tools
            - False: No tools
            - str: Use specific tool
            - list[str]: Allow specific tools
        tools: list of callables or import paths which can be used as tools
        model_settings: Additional model-specific settings

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If agent initialization fails
        TypeError: If result type doesn't match expected type
    """
    result_type = result_type or str  # type: ignore
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
                case FileEnvironment() | InlineEnvironment():
                    # Complete environment definition
                    update = {"environment": environment}
                    agent_config = agent_config.model_copy(update=update)
                # Config inherits from FileEnvironment, so order matters
                case Config():
                    # Direct runtime config
                    update = {"environment": InlineEnvironment.from_config(environment)}
                    agent_config = agent_config.model_copy(update=update)
                case _:
                    msg = f"Invalid environment type: {type(environment)}"
                    raise TypeError(msg)  # noqa: TRY301

        # Create agent with all settings
        async with Agent.open_agent(  # type: ignore[call-overload]
            agent_def,
            agent_name,
            result_type=result_type,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            model_settings=model_settings or {},
            retries=retries or 1,
        ) as agent:
            prompt = prompt if isinstance(prompt, list) else [prompt]
            result = await agent.run(*prompt)

            # Format output based on format
            match output_format:
                case "raw":
                    return result.data  # type: ignore[return-value]
                case "json":
                    if hasattr(result.data, "model_dump_json"):
                        return result.data.model_dump_json(indent=2)  # pyright: ignore
                    return str(result.data)
                case "yaml":
                    if hasattr(result.data, "model_dump_yaml"):
                        return result.data.model_dump_yaml()  # pyright: ignore
                    return str(result.data)
                case "text" | _:
                    return str(result.data)

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
    prompt: AnyPromptType | list[AnyPromptType],
    config: str | AgentsManifest,
    **kwargs: Any,
) -> Any:
    """Synchronous version of run_agent_pipeline.

    This is a convenience wrapper that runs the async pipeline in a new event loop.
    See run_agent_pipeline for full documentation.
    """
    fn = run_agent_pipeline(agent_name=agent_name, prompt=prompt, config=config, **kwargs)
    return asyncio.run(fn)


@overload
async def run_with_model(
    prompt: AnyPromptType | list[AnyPromptType],
    model: str | models.Model | models.KnownModelName,
    *,
    result_type: None = None,
    system_prompt: AnyPromptType | Sequence[AnyPromptType] = (),
    output_format: Literal["text", "json", "yaml"] = "text",
    model_settings: dict[str, Any] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[ToolType] | None = None,
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
) -> str: ...


@overload
async def run_with_model(
    prompt: AnyPromptType | list[AnyPromptType],
    model: str | models.Model | models.KnownModelName,
    *,
    result_type: type[T],
    system_prompt: AnyPromptType | Sequence[AnyPromptType] = (),
    output_format: OutputFormat = "raw",  # Allow any OutputFormat
    model_settings: dict[str, Any] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[ToolType] | None = None,
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
) -> T: ...


async def run_with_model(
    prompt: AnyPromptType | list[AnyPromptType],
    model: str | models.Model | models.KnownModelName,
    *,
    result_type: type[T] | None = None,
    system_prompt: AnyPromptType | Sequence[AnyPromptType] = (),
    output_format: OutputFormat = "text",
    model_settings: dict[str, Any] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[ToolType] | None = None,
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
        model_settings: Model-specific settings
        tool_choice: Control tool usage:
            - True: Allow all tools
            - False: No tools
            - str: Use specific tool
            - list[str]: Allow specific tools
        tools: list of callables or import paths which can be used as tools
        environment: Optional environment configuration
        error_handling: How to handle errors (raise/return/ignore)

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
    """
    # Create minimal manifest with optional response type
    responses = {}
    result_type = result_type or str  # type: ignore
    if result_type is not str:
        fields = {"result": ResponseField(type="str", description="Result")}
        r = InlineResponseDefinition(description="Default result type", fields=fields)
        responses["DefaultResult"] = r

    agent_environment = (
        InlineEnvironment.from_config(environment)
        if isinstance(environment, Config)
        else environment
    )
    cfg = AgentConfig(
        name="default",
        model=model,  # type: ignore
        system_prompts=[await to_prompt(system_prompt)],
        result_type="DefaultResult" if result_type else None,
        environment=agent_environment,
    )
    manifest = AgentsManifest[Any](responses=responses, agents={"default": cfg})
    return await run_agent_pipeline(
        "default",
        prompt,
        manifest,
        output_format=output_format,
        model_settings=model_settings,
        tool_choice=tool_choice,
        tools=tools,
        error_handling=error_handling,
        result_type=result_type,
    )


@overload
def run_with_model_sync(
    prompt: AnyPromptType | list[AnyPromptType],
    model: str | models.Model | models.KnownModelName,
    *,
    result_type: None = None,
    system_prompt: AnyPromptType | Sequence[AnyPromptType] = (),
    output_format: Literal["text", "json", "yaml"] = "text",
    model_settings: dict[str, Any] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[ToolType] | None = None,
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
) -> str: ...


@overload
def run_with_model_sync(
    prompt: AnyPromptType | list[AnyPromptType],
    model: str | models.Model | models.KnownModelName,
    *,
    result_type: type[T],
    system_prompt: AnyPromptType | Sequence[AnyPromptType] = (),
    output_format: Literal["raw"] = "raw",
    model_settings: dict[str, Any] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[ToolType] | None = None,
    environment: str | Config | AgentEnvironment | None = None,
    error_handling: ErrorHandling = "raise",
) -> T: ...


def run_with_model_sync(
    prompt: AnyPromptType | list[AnyPromptType],
    model: str | models.Model | models.KnownModelName,
    *,
    result_type: type[T] | None = None,
    system_prompt: AnyPromptType | Sequence[AnyPromptType] = (),
    output_format: OutputFormat = "text",
    model_settings: dict[str, Any] | None = None,
    tool_choice: bool | str | list[str] = True,
    tools: list[ToolType] | None = None,
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
        run_with_model(
            prompt=prompt,
            model=model,
            result_type=result_type,  # type: ignore
            system_prompt=system_prompt,
            output_format=output_format,  # type: ignore
            model_settings=model_settings,
            tool_choice=tool_choice,
            tools=tools,
            environment=environment,
            error_handling=error_handling,
        )
    )
