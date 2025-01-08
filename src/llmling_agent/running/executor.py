"""Function execution management."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llmling_agent.delegation import AgentPool, with_agents
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.running.discovery import AgentFunction


logger = get_logger(__name__)


class ExecutionError(Exception):
    """Raised when function execution fails."""


def discover_functions(path: str | Path) -> list[AgentFunction]:
    """Find all agent functions in a module.

    Args:
        path: Path to Python module file

    Returns:
        List of discovered agent functions

    Raises:
        ImportError: If module cannot be imported
        ValueError: If path is invalid
    """
    path_obj = Path(path)
    if not path_obj.exists():
        msg = f"Module not found: {path}"
        raise ValueError(msg)

    # Import module
    spec = importlib.util.spec_from_file_location(
        path_obj.stem,
        path_obj,
    )
    if not spec or not spec.loader:
        msg = f"Could not load module: {path}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find decorated functions
    functions = []
    for name, obj in inspect.getmembers(module):
        if hasattr(obj, "_agent_function"):
            functions.append(obj._agent_function)
            logger.debug("Discovered agent function: %s", name)

    return functions


def sort_functions(functions: list[AgentFunction]) -> list[AgentFunction]:
    """Sort functions by order and dependencies.

    Args:
        functions: Functions to sort

    Returns:
        Sorted list of functions

    Raises:
        ValueError: If there are circular dependencies
    """
    # First by explicit order
    ordered = sorted(
        functions,
        key=lambda f: (f.order or float("inf"), f.name),
    )

    # Then resolve dependencies
    result = []
    seen = set()
    in_progress = set()

    def add_function(func: AgentFunction):
        if func.name in seen:
            return
        if func.name in in_progress:
            msg = f"Circular dependency detected: {func.name}"
            raise ValueError(msg)

        in_progress.add(func.name)
        # Add dependencies first
        for dep in func.depends_on:
            dep_func = next(
                (f for f in ordered if f.name == dep),
                None,
            )
            if not dep_func:
                msg = f"Missing dependency {dep} for {func.name}"
                raise ValueError(msg)
            add_function(dep_func)

        result.append(func)
        in_progress.remove(func.name)
        seen.add(func.name)

    for func in ordered:
        add_function(func)

    return result


def group_parallel(
    sorted_funcs: list[AgentFunction],
) -> list[list[AgentFunction]]:
    """Group functions that can run in parallel."""
    if not sorted_funcs:
        return []

    # Group by dependency signature
    by_deps: dict[tuple[str, ...], list[AgentFunction]] = {}

    for func in sorted_funcs:
        # Use tuple of sorted deps as key for consistent grouping
        key = tuple(sorted(func.depends_on))
        if key not in by_deps:
            by_deps[key] = []
        by_deps[key].append(func)

    # Convert to list of groups, maintaining order
    groups = []
    seen_funcs: set[str] = set()

    for func in sorted_funcs:
        key = tuple(sorted(func.depends_on))
        if func.name not in seen_funcs:
            group = by_deps[key]
            groups.append(group)
            seen_funcs.update(f.name for f in group)

    logger.debug(
        "Grouped %d functions into %d groups: %s",
        len(sorted_funcs),
        len(groups),
        [[f.name for f in g] for g in groups],
    )
    return groups


async def execute_single(
    func: AgentFunction,
    pool: AgentPool,
    available_results: dict[str, Any],
    inputs: dict[str, Any] | None = None,
) -> tuple[str, Any]:
    """Execute a single function.

    Args:
        func: Function to execute
        pool: Agent pool for injection
        available_results: Results from previous functions
        inputs: Optional input overrides

    Returns:
        Tuple of (function name, result)

    Raises:
        ExecutionError: If execution fails
    """
    logger.debug("Executing %s", func.name)
    try:
        # Prepare inputs from defaults and provided inputs
        kwargs = func.default_inputs.copy()
        if inputs:
            kwargs.update(inputs)

        # Add results from dependencies
        for dep in func.depends_on:
            if dep not in available_results:
                msg = f"Missing result from {dep}"
                raise ExecutionError(msg)  # noqa: TRY301
            kwargs[dep] = available_results[dep]

        # Execute with agent injection
        wrapped = with_agents(pool)(func.func)
        result = await wrapped(**kwargs)
        logger.debug("%s returned: %s", func.name, result)

    except Exception as e:
        msg = f"Error executing {func.name}: {e}"
        raise ExecutionError(msg) from e
    else:
        return func.name, result


async def execute_functions(
    functions: list[AgentFunction],
    pool: AgentPool,
    inputs: dict[str, Any] | None = None,
    parallel: bool = False,
) -> dict[str, Any]:
    """Execute discovered functions in the right order."""
    logger.info(
        "Executing %d functions (parallel=%s)",
        len(functions),
        parallel,
    )
    results: dict[str, Any] = {}

    # Sort by order/dependencies
    sorted_funcs = sort_functions(functions)

    if parallel:
        # Group functions that can run in parallel
        groups = group_parallel(sorted_funcs)
        for i, group in enumerate(groups):
            logger.debug(
                "Executing parallel group %d/%d: %s",
                i + 1,
                len(groups),
                [f.name for f in group],
            )

            # Ensure previous results are available
            logger.debug("Available results: %s", sorted(results))

            # Run group in parallel
            tasks = [execute_single(func, pool, results, inputs) for func in group]
            group_results = await asyncio.gather(*tasks)

            # Update results after group completes
            results.update(dict(group_results))
            logger.debug("Group %d complete", i + 1)

            # Add small delay between groups to ensure timing separation
            if i < len(groups) - 1:
                await asyncio.sleep(0.02)  # 20ms between groups
    else:
        # Execute sequentially
        for func in sorted_funcs:
            name, result = await execute_single(func, pool, results, inputs)
            results[name] = result

    return results
