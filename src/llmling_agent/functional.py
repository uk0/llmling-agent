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

from llmling_agent import LLMlingAgent
from llmling_agent.environment.models import FileEnvironment, InlineEnvironment
from llmling_agent.log import get_logger
from llmling_agent.models import AgentsManifest, SystemPrompt


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling_agent.environment import AgentEnvironment


logger = get_logger(__name__)

# Type for the result
T = TypeVar("T")

# Type for output format
OutputFormat = Literal["text", "json", "yaml", "raw"]

# Type for error handling
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
    confirm_tools: set[str] | bool = False,
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
    confirm_tools: set[str] | bool = False,
    model_settings: dict[str, Any] | None = None,
) -> str: ...


@overload
async def run_agent_pipeline(
    agent_name: str,
    prompt: str | list[str] | SystemPrompt,
    config: str | AgentsManifest,
    *,
    stream: Literal[True],
    # ... other parameters ...
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
    confirm_tools: set[str] | bool = False,
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
        confirm_tools: Which tools need confirmation:
            - True: All tools
            - False: No tools
            - set[str]: Specific tools
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
                    agent_config = agent_config.model_copy(
                        update={"environment": environment}
                    )
                case Config():
                    # Direct runtime config
                    agent_config = agent_config.model_copy(
                        update={
                            "environment": {
                                "type": "inline",
                                "config": environment,
                            }
                        }
                    )
                case FileEnvironment() | InlineEnvironment():
                    # Complete environment definition
                    agent_config = agent_config.model_copy(
                        update={"environment": environment}
                    )
                case _:
                    msg = f"Invalid environment type: {type(environment)}"
                    raise TypeError(msg)  # noqa: TRY301

        # Update capabilities if provided
        if capabilities and agent_config.role in agent_def.roles:
            current = agent_def.roles[agent_config.role].model_dump()
            current.update(capabilities)
            agent_def.roles[agent_config.role] = agent_def.roles[
                agent_config.role
            ].model_copy(update=current)

        # Create agent with all settings
        async with LLMlingAgent[T].open_agent(
            agent_def,
            agent_name,
            model=model,  # type: ignore[arg-type]
            result_type=result_type,
            retries=retries,
            tool_choice=tool_choice,
            confirm_tools=confirm_tools,
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
                        async with await agent.run_stream(
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

    return asyncio.run(
        run_agent_pipeline(
            agent_name=agent_name,
            prompt=prompt,
            config=config,
            **kwargs,
        )
    )
