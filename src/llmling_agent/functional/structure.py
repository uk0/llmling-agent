"""High-level pipeline functions for agent execution."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, get_args

from llmling import Config

from llmling_agent.agent.agent import LLMlingAgent
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_ai.agent import models

logger = get_logger(__name__)


async def get_structured[T](
    prompt: str,
    response_type: type[T],
    model: models.Model | models.KnownModelName,
    *,
    system_prompt: str | None = None,
    max_retries: int = 3,
    error_handler: Callable[[Exception], T | None] | None = None,
) -> T:
    """Get structured output from LLM using function calling.

    This function creates a temporary agent that uses the class constructor
    as a tool to generate structured output. It handles:
    - Type conversion from Python types to JSON schema
    - Constructor parameter validation
    - Error handling with optional recovery

    Args:
        prompt: The prompt to send to the LLM
        response_type: The type to create (class with typed constructor)
        model: model to use
        system_prompt: Optional system instructions
        max_retries: Max attempts for parsing (default: 3)
        error_handler: Optional error handler for recovery

    Returns:
        Instance of response_type

    Example:
        ```python
        class TaskResult:
            '''Analysis result for a task.'''
            def __init__(
                self,
                success: bool,
                message: str,
                due_date: datetime | None = None
            ):
                self.success = success
                self.message = message
                self.due_date = due_date

        result = await get_structured(
            "Analyze task: Deploy monitoring",
            TaskResult,
            system_prompt="You analyze task success"
        )
        print(f"Success: {result.success}")
        ```

    Raises:
        TypeError: If response_type is not a valid type
        ValueError: If constructor schema cannot be created
        Exception: If LLM call fails and no error_handler recovers
    """
    # Create constructor tool
    from py2openai import create_constructor_schema

    schema = create_constructor_schema(response_type).model_dump_openai()["function"]

    async def construct(**kwargs: Any) -> T:
        """Construct instance from LLM-provided arguments."""
        return response_type(**kwargs)

    async with LLMlingAgent[Any, T].open(
        result_type=response_type,
        model=model,
        system_prompt=system_prompt or [],
        name="structured",
        retries=max_retries,
    ) as agent:
        # Register constructor as only tool
        tool = agent._pydantic_agent.tool_plain(construct)
        tool.__name__ = schema["name"]
        tool.__doc__ = schema["description"]

        try:
            result = await agent.run(prompt)
        except Exception as e:
            if error_handler and (err_result := error_handler(e)):
                return err_result
            raise
        else:
            return result.data


async def get_structured_multiple[T](
    prompt: str,
    target: type[T],
    model: models.Model | models.KnownModelName,
) -> list[T]:
    """Extract multiple structured instances from text."""
    instances: list[T] = []
    import inspect

    async def add_instance(**kwargs: Any) -> str:
        """Add an extracted instance."""
        instance = target(**kwargs)
        instances.append(instance)
        return f"Added {instance}"

    # Get class and init documentation
    class_doc = inspect.getdoc(target) or f"A {target.__name__}"
    init_doc = inspect.getdoc(target.__init__) or "Create a new instance."
    type_info = (
        f"\nInstance type information:\n{class_doc}\n\nInitialization:\n{init_doc}"
    )

    system_prompts = [
        f"You are an expert at extracting {target.__name__} instances from text.",
        "IMPORTANT: You must use the provided add_instance function for EACH instance.",
        "DO NOT just describe what you found - you must CALL add_instance.",
        "",
        "Process:",
        "1. Find an instance",
        "2. Call add_instance with its properties",
        "3. Continue until no more instances can be found",
        "",
        "Example:",
        "I found John Smith, calling add_instance...",
        "[call add_instance with first_name='John', last_name='Smith']",
        "Looking for more...",
        "",
        type_info,
    ]

    async with LLMlingAgent[Any, Any].open(
        Config(),
        model=model,
        system_prompt=system_prompts,
        name="structured",
        tool_choice=f"add_{target.__name__}",  # Force using our tool
    ) as agent:
        agent.tools.register_tool(add_instance, enabled=True, source="dynamic")
        logger.debug("Running extraction with prompt: %s", prompt)
        prompt = f"Extract ALL {target.__name__} instances from this text: {prompt}"
        await agent.run(prompt)
        logger.debug("Found %d instances", len(instances))
        return instances


async def pick_one[T](
    prompt: str,
    options: type[T | Enum] | list[T],
    model: models.Model | models.KnownModelName,
) -> T:
    """Pick one option from a list of choices."""
    instances: dict[str, T] = {}

    # Create mapping and descriptions
    if isinstance(options, type):
        if issubclass(options, Enum):
            choices = {e.name: (e.value, str(e.value)) for e in options}
        else:
            literal_opts = get_args(options)
            choices = {str(opt): (opt, str(opt)) for opt in literal_opts}
    else:  # List
        choices = {str(i): (opt, repr(opt)) for i, opt in enumerate(options)}

    async def select_option(option: Literal[tuple(choices.keys())]) -> str:  # type: ignore
        """Pick one of the available options.

        Args:
            option: Which option to pick
        """
        instances["selected"] = choices[option][0]
        return f"Selected: {option}"

    # Add options to docstring
    docs = "\n".join(f"- {k}: {desc}" for k, (_, desc) in choices.items())
    assert select_option.__doc__
    select_option.__doc__ += f"\nOptions:\n{docs}"

    async with LLMlingAgent[Any, Any].open(
        Config(),
        model=model,
        system_prompt="Select the most appropriate option based on the context.",
    ) as agent:
        agent.tools.register_tool(select_option, enabled=True)
        agent._tool_manager.tool_choice = "select_option"

        await agent.run(prompt)
        return instances["selected"]
