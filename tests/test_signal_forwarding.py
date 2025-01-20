from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from llmling_agent import AgentPool, AgentsManifest


if TYPE_CHECKING:
    from pathlib import Path


BASIC_FORWARDING = """\
agents:
    agent1:
        model: openai:gpt-4o-mini
        name: TestAgent 1
        connections:
            - type: agent
              name: agent2

    agent2:
        model: openai:gpt-4o-mini
        name: TestAgent 2
        connections:
            - type: agent
              name: agent3

    agent3:
        model: openai:gpt-4o-mini
        name: TestAgent 3
"""


INVALID_FORWARD = """\
agents:
    agent1:
        model: openai:gpt-4o-mini
        name: TestAgent
        connections:
            - type: agent
              name: nonexistent
"""


PARTIAL_FORWARDING = """\
agents:
    agent1:
        model: openai:gpt-4o-mini
        name: TestAgent 1
        connections:
            - type: agent
              name: agent2

    agent2:
        model: openai:gpt-4o-mini
        name: TestAgent 2
"""


@pytest.fixture
def basic_config(tmp_path: Path) -> Path:
    """Create a temporary config file with basic forwarding setup."""
    config_file = tmp_path / "agents.yml"
    config_file.write_text(BASIC_FORWARDING)
    return config_file


@pytest.fixture
def partial_config(tmp_path: Path) -> Path:
    """Create a temporary config file with partial forwarding setup."""
    config_file = tmp_path / "partial.yml"
    config_file.write_text(PARTIAL_FORWARDING)
    return config_file


@pytest.fixture
def invalid_config(tmp_path: Path) -> Path:
    """Create a temporary config file with invalid forwarding setup."""
    config_file = tmp_path / "invalid.yml"
    config_file.write_text(INVALID_FORWARD)
    return config_file


async def test_agent_forwarding(basic_config: Path):
    """Test that messages get forwarded through the agent chain."""
    manifest = AgentsManifest[Any].from_file(basic_config)

    async with AgentPool[None](manifest) as pool:
        agent1 = pool.get_agent("agent1")
        agent2 = pool.get_agent("agent2")
        agent3 = pool.get_agent("agent3")

        responded_agents = set()
        received_messages = []

        def record_response(agent_name: str):
            def callback(message):
                responded_agents.add(agent_name)
                received_messages.append(f"{agent_name}: {message.content}")
                print(f"Message from {agent_name}: {message.content}")

            return callback

        agent1.message_sent.connect(record_response("agent1"))
        agent2.message_sent.connect(record_response("agent2"))
        agent3.message_sent.connect(record_response("agent3"))

        await agent1.run("test")
        # Wait for all forwarded messages to be processed
        await agent1.complete_tasks()
        await agent2.complete_tasks()
        await agent3.complete_tasks()

        print("Received messages:", received_messages)
        assert responded_agents == {"agent1", "agent2", "agent3"}


async def test_partial_chain(partial_config: Path):
    """Test forwarding with only some agents loaded."""
    manifest = AgentsManifest[Any].from_file(partial_config)

    async with AgentPool[None](manifest) as pool:
        agent1 = pool.get_agent("agent1")
        agent2 = pool.get_agent("agent2")

        responded_agents = set()
        agent1.message_sent.connect(lambda _: responded_agents.add("agent1"))
        agent2.message_sent.connect(lambda _: responded_agents.add("agent2"))

        await agent1.run("test")
        await agent2.complete_tasks()
        assert responded_agents == {"agent1", "agent2"}


async def test_invalid_forward_target(invalid_config: Path):
    """Test error when forwarding to non-existent agent."""
    manifest = AgentsManifest[Any].from_file(invalid_config)

    with pytest.raises(ValueError, match="Forward target.*not found"):
        async with AgentPool[None](manifest):
            pass
