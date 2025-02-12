import asyncio

from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent import Agent
from llmling_agent.observability import registry, track_action
from llmling_agent_observability.mock_provider import MockProvider


@pytest.fixture
def mock_provider():
    provider = MockProvider()
    old_providers = registry.providers
    registry.configure_provider(provider)
    yield provider
    registry.providers = old_providers


def test_action_tracking(mock_provider: MockProvider):
    @track_action("test_action")
    async def some_function():
        return "result"

    # Run the function
    _result = asyncio.run(some_function())

    # Verify the action was tracked
    assert any(
        call.call_type == "wrap_action" and call.name == "test_action"
        for call in mock_provider.calls
    )


def test_span_tracking(mock_provider: MockProvider):
    with mock_provider.span("test_span"):
        pass

    assert any(
        call.call_type == "span" and call.name == "test_span"
        for call in mock_provider.calls
    )


def test_agent_run_action_tracking(mock_provider: MockProvider):
    # Create agent with test model
    model = TestModel(custom_result_text="Test response")
    agent = Agent[None](name="test-agent", model=model)
    _result = agent.run_sync("Test prompt")
    # Verify that _run method was tracked as an action
    tracked_actions = [
        call
        for call in mock_provider.calls
        if call.call_type == "wrap_action"
        and call.name == "Calling Agent.run: {prompts}:"
    ]
    assert len(tracked_actions) == 1
    assert tracked_actions[0].kwargs["msg_template"] == "Calling Agent.run: {prompts}:"


def test_multiple_providers():
    provider1 = MockProvider()
    provider2 = MockProvider()

    registry.configure_provider(provider1)
    registry.configure_provider(provider2)

    @track_action("test_action")
    async def some_function():
        return "result"

    _result = asyncio.run(some_function())

    # Verify both providers tracked the action
    for provider in (provider1, provider2):
        assert any(
            call.call_type == "wrap_action" and call.name == "test_action"
            for call in provider.calls
        )


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
