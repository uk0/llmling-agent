"""Test basic connection behavior and statistics."""

from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent import Agent


async def test_basic_single_connection():
    """Test basic message forwarding between two agents."""
    async with (
        Agent[str](model="test", name="llmling-agent") as source,
        Agent[str](model="test") as target,
    ):
        # Create explicit connection
        talk = source.pass_results_to(target)

        # Send a message
        await source.run("test message")

        # Check stats
        stats = talk.stats  # Now using .stats property
        assert stats.message_count == 1
        assert stats.source_name == source.name
        assert target.name in stats.target_names


async def test_multiple_targets():
    """Test forwarding to multiple targets with stats tracking."""
    async with (
        Agent[str](model="test") as source,
        Agent[str](model="test") as target1,
        Agent[str](model="test") as target2,
    ):
        # Create separate connections
        talk1 = source.pass_results_to(target1)
        talk2 = source.pass_results_to(target2)

        # Send two messages
        await source.run("message 1")  # One message through each talk
        await source.run("message 2")  # One more through each talk

        # Each talk processes both messages (results are passed to both)
        assert talk1.stats.message_count == 2  # noqa: PLR2004
        assert talk2.stats.message_count == 2  # noqa: PLR2004

        # Manager aggregates all talks
        group_stats = source.connections.stats
        assert group_stats.num_connections == 2  # noqa: PLR2004
        assert group_stats.message_count == 4  # 2 talks * 1 message each  # noqa: PLR2004


async def test_connection_filtering():
    """Test connection filtering with when() condition."""
    async with (
        Agent[str](model="test") as source,
        Agent[str](model="test") as target,
    ):
        # Only forward messages containing "important"
        talk = source.pass_results_to(target)
        talk.when(lambda msg, _stats: "important" in msg.content)

        # First message with default test model response
        await source.run("first message")

        # Second message with custom response
        model = TestModel(custom_result_text="important response from model")
        source.set_model(model)
        await source.run("second message")

        assert talk.stats.message_count == 1  # Only the message containing "importan


async def test_disconnect():
    """Test disconnecting agents."""
    async with (
        Agent[str](model="test") as source,
        Agent[str](model="test") as target,
    ):
        talk = source.pass_results_to(target)

        # Send message while connected
        await source.run("message 1")
        assert talk.stats.message_count == 1

        # Disconnect and send another
        source.stop_passing_results_to(target)
        await source.run("message 2")
        assert talk.stats.message_count == 1  # Still just one message


async def test_token_tracking():
    """Test token counting in connection stats."""
    async with (
        Agent[str](model="test") as source,
        Agent[str](model="test") as target,
    ):
        talk = source.pass_results_to(target)
        await source.run("test message")

        assert talk.stats.token_count > 0  # Actual number depends on model


async def test_group_stats_aggregation():
    """Test GroupStats aggregation of multiple connections."""
    async with (
        Agent[str](model="test", name="source") as source,
        Agent[str](model="test", name="target1") as target1,
        Agent[str](model="test", name="target2") as target2,
    ):
        # Create team connection
        team = target1 & target2
        team_talk = source.pass_results_to(team)

        # Send message
        await source.run("test message")

        # Check group stats
        group_stats = team_talk.stats
        assert group_stats.num_connections == 2  # noqa: PLR2004
        assert group_stats.message_count == 2  # noqa: PLR2004 # One message to two targets
        assert len(group_stats.source_names) == 1
        assert len(group_stats.target_names) == 2  # noqa: PLR2004
        assert group_stats.start_time is not None
        assert group_stats.last_message_time is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
