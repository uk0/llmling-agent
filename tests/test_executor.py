from __future__ import annotations

from typing import Any

import pytest

from llmling_agent import Agent  # noqa: TC001
from llmling_agent.running.discovery import agent_function
from llmling_agent.running.executor import (
    ExecutionError,
    execute_functions,
    group_parallel,
    sort_functions,
)


async def test_function_sorting():
    """Test function sorting by dependencies and definition order."""

    @agent_function
    async def first(): ...

    @agent_function(depends_on="first")
    async def second(): ...

    @agent_function(depends_on="second")
    async def third(): ...

    funcs = [second, third, first]
    sorted_funcs = sort_functions([f._agent_function for f in funcs])
    assert [f.name for f in sorted_funcs] == ["first", "second", "third"]


async def test_circular_dependency():
    """Test detection of circular dependencies."""

    @agent_function(depends_on="b")
    async def a(): ...

    @agent_function(depends_on="a")
    async def b(): ...

    with pytest.raises(ValueError, match="Circular dependency"):
        sort_functions([a._agent_function, b._agent_function])


async def test_parallel_grouping():
    """Test grouping of parallel functions."""

    @agent_function
    async def first(): ...

    @agent_function(depends_on="first")
    async def second_a(): ...

    @agent_function(depends_on="first")
    async def second_b(): ...

    @agent_function(depends_on=["second_a", "second_b"])
    async def third(): ...

    sorted_funcs = sort_functions([
        f._agent_function for f in [first, second_a, second_b, third]
    ])
    groups = group_parallel(sorted_funcs)

    assert len(groups) == 3  # noqa: PLR2004
    assert [f.name for f in groups[0]] == ["first"]
    assert {f.name for f in groups[1]} == {"second_a", "second_b"}
    assert [f.name for f in groups[2]] == ["third"]


async def test_execution_order(pool):
    """Test function execution order."""
    executed = []

    @agent_function
    async def first(agent1: Agent[None]):
        executed.append("first")
        return "first"

    @agent_function(depends_on="first")
    async def second(agent1: Agent[None], first: str):  # Gets result from first:
        assert first == "first"
        executed.append("second")
        return "second"

    funcs = [second._agent_function, first._agent_function]  # type: ignore
    results = await execute_functions(funcs, pool)

    assert executed == ["first", "second"]
    assert results == {"first": "first", "second": "second"}


async def test_parallel_execution(pool):
    """Test parallel function execution."""
    import asyncio

    # Track execution times
    start_times: dict[str, float] = {}
    end_times: dict[str, float] = {}

    @agent_function
    async def first(agent1: Agent[None]):
        start_times["first"] = asyncio.get_event_loop().time()
        await asyncio.sleep(0.1)
        end_times["first"] = asyncio.get_event_loop().time()
        return "first"

    @agent_function(depends_on="first")
    async def second_a(agent1: Agent[None], first: str):
        start_times["second_a"] = asyncio.get_event_loop().time()
        await asyncio.sleep(0.1)
        end_times["second_a"] = asyncio.get_event_loop().time()
        return "second_a"

    @agent_function(depends_on="first")
    async def second_b(agent1: Agent[None], first: str):
        start_times["second_b"] = asyncio.get_event_loop().time()
        await asyncio.sleep(0.1)
        end_times["second_b"] = asyncio.get_event_loop().time()
        return "second_b"

    funcs = [f._agent_function for f in [first, second_a, second_b]]

    await execute_functions(funcs, pool, parallel=True)

    # Add small epsilon for float comparison
    epsilon = 0.001  # 1ms buffer

    # First should complete before second_a/second_b start
    assert end_times["first"] + epsilon < start_times["second_a"]
    assert end_times["first"] + epsilon < start_times["second_b"]

    # second_a and second_b should overlap
    assert start_times["second_a"] < end_times["second_b"]
    assert start_times["second_b"] < end_times["second_a"]


async def test_input_handling(pool):
    """Test handling of inputs and defaults."""

    @agent_function
    async def func(agent1: Agent[None], required: str, optional: int = 42):
        return f"{required}:{optional}"

    # With default
    inputs: dict[str, Any] = {"required": "test"}
    result = await execute_functions([func._agent_function], pool, inputs=inputs)
    assert result["func"] == "test:42"

    # With override
    inputs = {"required": "test", "optional": 100}
    result = await execute_functions([func._agent_function], pool, inputs=inputs)
    assert result["func"] == "test:100"

    # Missing required
    with pytest.raises(ExecutionError):
        await execute_functions([func._agent_function], pool)  # type: ignore
