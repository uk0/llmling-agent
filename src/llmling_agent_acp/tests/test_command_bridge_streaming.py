"""Tests for streaming command bridge functionality."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from slashed import CommandStore

from llmling_agent import Agent, AgentPool
from llmling_agent_acp.command_bridge import ACPCommandBridge, ACPOutputWriter
from llmling_agent_acp.converters import to_agent_text_notification
from llmling_agent_acp.session import ACPSession


@pytest.mark.asyncio
async def test_acp_output_writer():
    """Test that ACPOutputWriter sends updates immediately to session."""
    # Mock session with client
    mock_session = AsyncMock(spec=ACPSession)
    mock_session.session_id = "test_session"
    mock_session.client = AsyncMock()

    received_updates = []

    async def capture_update(update):
        received_updates.append(update)

    mock_session.client.session_update.side_effect = capture_update

    writer = ACPOutputWriter(mock_session)

    # Test sending multiple messages
    await writer.print("First message")
    await writer.print("Second message")

    # Verify updates were sent immediately
    expected_update_count = 2
    assert len(received_updates) == expected_update_count

    # Verify content matches expected session updates
    expected_first = to_agent_text_notification("First message", "test_session")
    expected_second = to_agent_text_notification("Second message", "test_session")

    assert received_updates[0] == expected_first
    assert received_updates[1] == expected_second


@pytest.mark.asyncio
async def test_command_bridge_immediate_execution():
    """Test that command execution sends updates immediately."""

    def simple_callback(message: str) -> str:
        return f"Response: {message}"

    # Set up agent and session
    agent = Agent(name="test_agent", provider=simple_callback)
    agent_pool = AgentPool[None]()
    agent_pool.register("test_agent", agent)

    # Create command store and bridge
    command_store = CommandStore()
    command_bridge = ACPCommandBridge(command_store)

    # Mock session with client
    mock_session = AsyncMock(spec=ACPSession)
    mock_session.session_id = "test_session"
    mock_session.agent = AsyncMock()
    mock_session.agent.context = None
    mock_session.client = AsyncMock()

    # Capture updates sent to client
    sent_updates = []

    async def capture_update(update):
        sent_updates.append(update)

    mock_session.client.session_update.side_effect = capture_update

    # Test command execution - now it's not an async generator
    await command_bridge.execute_slash_command("/help", mock_session)

    # Verify updates were sent immediately to client
    assert len(sent_updates) > 0
    assert all(update.session_id == "test_session" for update in sent_updates)


@pytest.mark.asyncio
async def test_immediate_send_with_slow_command():
    """Test immediate sending works with commands that produce output over time."""

    # Create a command that outputs multiple lines with delays
    async def slow_command_func(ctx, args, kwargs):
        await ctx.output.print("Starting task...")
        await asyncio.sleep(0.01)  # Small delay
        await ctx.output.print("Processing...")
        await asyncio.sleep(0.01)  # Small delay
        await ctx.output.print("Completed!")

    # Set up command store
    command_store = CommandStore()
    command_store.add_command(
        name="slow", fn=slow_command_func, description="A slow command for testing"
    )

    command_bridge = ACPCommandBridge(command_store)

    # Mock session with client
    mock_session = AsyncMock(spec=ACPSession)
    mock_session.session_id = "test_session"
    mock_session.agent = AsyncMock()
    mock_session.agent.context = None
    mock_session.client = AsyncMock()

    # Collect updates with timestamps to verify immediate sending
    updates_with_time = []
    start_time = asyncio.get_event_loop().time()

    async def capture_with_time(update):
        current_time = asyncio.get_event_loop().time()
        updates_with_time.append((update, current_time - start_time))

    mock_session.client.session_update.side_effect = capture_with_time

    # Execute command
    await command_bridge.execute_slash_command("/slow", mock_session)

    # Verify we got multiple updates
    min_expected_updates = 3
    assert len(updates_with_time) >= min_expected_updates

    # Verify updates came at different times (immediate sending behavior)
    times = [time for _, time in updates_with_time]
    assert times[1] > times[0]  # Second update came after first
    assert times[2] > times[1]  # Third update came after second

    # Verify session IDs are correct
    for update, _ in updates_with_time:
        assert update.session_id == "test_session"


@pytest.mark.asyncio
async def test_immediate_send_error_handling():
    """Test that errors in commands are properly sent immediately."""

    async def failing_command(ctx, args, kwargs):
        await ctx.output.print("Starting...")
        msg = "Command failed!"
        raise ValueError(msg)

    command_store = CommandStore()
    command_store.add_command(
        name="fail", fn=failing_command, description="A failing command"
    )

    command_bridge = ACPCommandBridge(command_store)

    # Mock session with client
    mock_session = AsyncMock(spec=ACPSession)
    mock_session.session_id = "test_session"
    mock_session.agent = AsyncMock()
    mock_session.agent.context = None
    mock_session.client = AsyncMock()

    # Collect all updates
    sent_updates = []

    async def capture_update(update):
        sent_updates.append(update)

    mock_session.client.session_update.side_effect = capture_update

    # Execute failing command
    await command_bridge.execute_slash_command("/fail", mock_session)

    # Should get the initial output plus error message
    min_expected_updates = 2
    assert len(sent_updates) >= min_expected_updates

    # Check that we got both normal output and error
    update_contents = [
        update.update.content.text
        for update in sent_updates
        if hasattr(update.update, "content") and hasattr(update.update.content, "text")
    ]

    # Should contain both the initial message and error
    content_text = " ".join(update_contents)
    assert "Starting..." in content_text
    assert "Command error:" in content_text
