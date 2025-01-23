from __future__ import annotations

import asyncio

import pytest

from llmling_agent import Agent


@pytest.mark.asyncio
async def test_agent_piping():
    # Use explicit names for all agents
    agent1 = Agent[None].from_callback(lambda x: f"model: {x}", name="agent1")
    transform1 = Agent[None].from_callback(
        lambda x: f"transform1: {x}", name="transform1"
    )
    transform2 = Agent[None].from_callback(
        lambda x: f"transform2: {x}", name="transform2"
    )

    pipeline = agent1 | transform1 | transform2
    result = await pipeline.run("test")

    # Check message flow
    assert result[0].message
    assert result[1].message
    assert result[2].message
    assert result[0].message.data == "model: test"
    assert result[1].message.data == "transform1: model: test"
    assert result[2].message.data == "transform2: transform1: model: test"
    assert pipeline.stats.stats.message_count == 3  # noqa: PLR2004


@pytest.mark.asyncio
async def test_agent_piping_with_monitoring():
    def callback(text: str) -> str:
        return f"model: {text}"

    agent1 = Agent[None].from_callback(callback, name="agent1")

    async def transform(text: str) -> str:
        await asyncio.sleep(0.1)  # Add a delay to ensure monitoring catches it
        return f"transform: {text}"

    pipeline = agent1 | transform

    # Get stats object directly from run_in_background
    stats = pipeline.run_in_background("test")

    # Monitor progress
    updates = []
    while pipeline.is_running:
        updates.append(len(stats))  # Track number of active connections
        await asyncio.sleep(0.01)

    _result = await pipeline.wait()
    assert updates  # Last update should show no active agents


@pytest.mark.asyncio
async def test_agent_piping_errors():
    agent1 = Agent[None].from_callback(lambda x: f"model: {x}", name="agent1")
    failing = Agent[None].from_callback(
        lambda x: exec('raise ValueError("Transform error")'),  # type: ignore
        name="failing_transform",
    )

    pipeline = agent1 | failing
    result = await pipeline.run("test")

    assert result[0].success
    assert result[0].message
    assert result[0].message.data == "model: test"
    assert not result[1].success
    assert result[1].error is not None
    assert "Transform error" in result[1].error

    assert pipeline.stats.errors


@pytest.mark.asyncio
async def test_agent_piping_async():
    async def model_callback(text: str) -> str:
        return f"model: {text}"

    agent1 = Agent[None].from_callback(model_callback, name="agent1")

    async def transform(text: str) -> str:
        return f"transform: {text}"

    pipeline = agent1 | transform
    result = await pipeline.run("test")

    assert result.successful
    assert len(result) == 2  # noqa: PLR2004
    assert result[0].message
    assert result[1].message
    assert result[0].message.data == "model: test"
    assert result[1].message.data == "transform: model: test"
