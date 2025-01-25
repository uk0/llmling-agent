"""Tests for Team execution."""

from __future__ import annotations

import pytest

from llmling_agent.agent.agent import Agent
from llmling_agent.delegation import AgentPool
from llmling_agent.delegation.team import Team
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


@pytest.mark.asyncio
async def test_nested_teams():
    """Test nesting Teams and TeamRuns inside each other."""
    async with AgentPool() as pool:
        # Create basic agents
        a1 = await pool.add_agent("a1", model="test")
        a2 = await pool.add_agent("a2", model="test")
        a3 = await pool.add_agent("a3", model="test")
        a4 = await pool.add_agent("a4", model="test")

        # Case 1: Team inside TeamRun
        team = a1 & a2  # Team of two agents
        execution = team | a3  # TeamRun with Team + Agent
        result = await execution.run("test message")
        assert isinstance(result, ChatMessage)
        # Team's messages should be in the chain
        assert (
            len(execution.stats.stats.messages) == 2  # noqa: PLR2004
        )  # Team(a1+a2) + a3

        # Case 2: TeamRun inside Team
        sequential = a1 | a2  # TeamRun
        parallel_team = Team([sequential, a3, a4])  # Team containing TeamRun + Agents

        result = await parallel_team.run("test message")
        assert isinstance(result, ChatMessage)
        # Should have all messages
        assert (
            len(parallel_team.stats.stats.messages) == 3  # noqa: PLR2004
        )  # TeamRun(a1+a2) + a3 + a4

        # # Test streaming with nested Team
        # async with execution.run_stream("test message") as stream:
        #     chunks = [chunk async for chunk in stream.stream()]
        #     assert chunks  # Should get chunks from all agents

        # Test iteration with nested TeamRun
        messages = [msg async for msg in parallel_team.run_iter("test message")]
        assert len(messages) == 3  # Should get all messages  # noqa: PLR2004


# @pytest.mark.asyncio
# async def test_team_operators():
#     """Test team combination operators (& and |)."""
#     async with AgentPool() as pool:
#         # Create basic agents
#         a1 = await pool.add_agent("a1", model="test")
#         a2 = await pool.add_agent("a2", model="test")
#         a3 = await pool.add_agent("a3", model="test")
#         a4 = await pool.add_agent("a4", model="test")

#         # Test parallel combinations (&)
#         # Simple agent combinations
#         team1 = a1 & a2
#         assert isinstance(team1, Team)
#         assert len(team1.agents) == 2
#         assert list(team1.agents) == [a1, a2]

#         # Adding agent to team
#         team2 = team1 & a3
#         assert isinstance(team2, Team)
#         assert len(team2.agents) == 3
#         assert list(team2.agents) == [a1, a2, a3]

#         # Combining teams - should flatten
#         other_team = a3 & a4
#         combined = team1 & other_team
#         assert isinstance(combined, Team)
#         assert len(combined.agents) == 4
#         assert list(combined.agents) == [a1, a2, a3, a4]

#         # Test sequential combinations (|)
#         # Simple agent combinations
#         seq1 = a1 | a2
#         assert isinstance(seq1, TeamRun)
#         assert len(seq1.agents) == 2
#         assert list(seq1.agents) == [a1, a2]

#         # Adding to TeamRun - should extend
#         seq2 = seq1 | a3
#         assert seq2 is seq1  # Same TeamRun instance
#         assert len(seq1.agents) == 3
#         assert list(seq1.agents) == [a1, a2, a3]

#         # Complex combinations
#         team3 = a1 & a2  # parallel
#         seq3 = a3 | a4  # sequential

#         # TeamRun with Team member
#         combined_1 = team3 | seq3
#         assert isinstance(combined_1, TeamRun)
#         assert len(combined_1.agents) == 2
#         assert combined_1.agents[0] is team3
#         assert isinstance(combined_1.agents[1], TeamRun)

#         # Team with TeamRun member
#         combined_2 = Team([team3, seq3])
#         assert isinstance(combined_2, Team)
#         assert len(combined_2.agents) == 2
#         assert combined_2.agents[0] is team3
#         assert isinstance(combined_2.agents[1], TeamRun)

#         # Test actual execution
#         result = await combined_1.run("test")
#         assert isinstance(result, ChatMessage)
#         # All agents should have executed
#         assert len(combined_1.stats.stats.messages) == 4

#         result = await combined_2.run("test")
#         assert isinstance(result, ChatMessage)
#         # All agents should have executed
#         assert len(combined_2.stats.stats.messages) == 4


if __name__ == "__main__":
    pytest.main([__file__])
