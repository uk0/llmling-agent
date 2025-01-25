"""Tests for Team execution."""

from __future__ import annotations

import pytest

from llmling_agent.agent.agent import Agent
from llmling_agent.delegation import AgentPool
from llmling_agent.models.messages import ChatMessage


@pytest.mark.asyncio
async def test_team_parallel_execution():
    """Test that team runs all agents in parallel and collects responses."""
    async with AgentPool[None]() as pool:
        # Create three agents that append their name to input
        a1 = await pool.add_agent("a1", system_prompt="Append 'a1'", model="test")
        a2 = await pool.add_agent("a2", system_prompt="Append 'a2'", model="test")
        a3 = await pool.add_agent("a3", system_prompt="Append 'a3'", model="test")

        team = pool.create_team([a1, a2, a3])
        result = await team.execute("test")

        # Check that we got responses from all agents
        assert len(result) == 3  # noqa: PLR2004
        agent_names = {r.agent_name for r in result}
        assert agent_names == {"a1", "a2", "a3"}

        # Check that stats were collected
        assert len(team.stats.stats.messages) == 3  # noqa: PLR2004
        assert all(isinstance(msg, ChatMessage) for msg in team.stats.stats.messages)


@pytest.mark.asyncio
async def test_team_shared_prompt():
    """Test that shared prompt is prepended to individual prompts."""
    async with AgentPool[None]() as pool:
        # Create agents that echo their input
        def echo(prompt: str) -> str:
            return prompt

        a1 = pool.get_agent(Agent.from_callback(echo, name="a1"))
        a2 = pool.get_agent(Agent.from_callback(echo, name="a2"))

        # Create team with shared prompt
        team = pool.create_team(
            [a1, a2],
            shared_prompt="Common instruction: ",
        )
        result = await team.execute("specific task")

        # Each agent should get both prompts
        assert len(result) == 2  # noqa: PLR2004
        for response in result:
            assert response.message
            assert "Common instruction" in str(response.message.content)
            assert "specific task" in str(response.message.content)


if __name__ == "__main__":
    pytest.main([__file__])
