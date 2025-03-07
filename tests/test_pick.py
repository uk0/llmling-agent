import pytest

from llmling_agent import Agent


@pytest.mark.asyncio
async def test_pick_from_options():
    """Test picking from a list of options."""
    # Create agent for making decisions
    decider = Agent[None](
        model="openai:gpt-4o-mini",
        system_prompt="You are an expert at making clear decisions.",
    )

    # Test picking from simple options
    options = ["A", "B", "C"]

    decision = await decider.talk.pick(options, task="Pick A and give a random reason!")

    assert decision.selection in options
    assert len(decision.reason) > 0


@pytest.mark.asyncio
async def test_pick_from_agents():
    """Test picking from a team of agents."""
    # Create a team of specialized agents
    analyzer = Agent[None](
        name="code_analyzer",
        model="openai:gpt-4o-mini",
        description="Specializes in code analysis and best practices",
    )
    reviewer = Agent[None](
        name="security_expert",
        model="openai:gpt-4o-mini",
        description="Focuses on security vulnerabilities",
    )
    team = [analyzer, reviewer]

    # Create decision maker
    decider = Agent[None](
        model="openai:gpt-4o-mini",
        system_prompt="You are an expert at delegating tasks.",
    )

    # Test agent selection
    decision = await decider.talk.pick(
        team,
        task="We found potential SQL injection vulnerabilities. Who should investigate?",
    )

    assert decision.selection in team
    assert decision.selection.name == "security_expert"  # Should pick security expert
    assert "security" in decision.reason.lower()


@pytest.mark.asyncio
async def test_pick_multiple():
    """Test picking multiple options with constraints."""
    decider = Agent[None](model="openai:gpt-4o-mini")

    decision = await decider.talk.pick_multiple(
        ["A", "B", "C"],
        task="Pick A and B. Always pick both!.",
        min_picks=2,
        max_picks=2,
    )

    assert decision.selections == ["A", "B"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
