"""Integration tests for prompt commands."""

from __future__ import annotations

from typing import Any

from llmling import Config, RuntimeConfig
from llmling.prompts.models import PromptMessage, PromptParameter, StaticPrompt
import pytest

from llmling_agent import LLMlingAgent
from llmling_agent.chat_session import AgentChatSession
from llmling_agent.chat_session.output import DefaultOutputWriter
from llmling_agent.commands.base import CommandContext
from llmling_agent.commands.builtin.prompts import prompt_cmd


@pytest.fixture
def runtime_config() -> Config:
    """Create a RuntimeConfig with test prompts."""
    # Create prompt messages
    greet_message = PromptMessage(role="system", content="Hello {name}!")

    analyze_messages = [
        PromptMessage(role="system", content="Analyzing {data}..."),
        PromptMessage(role="user", content="Please check {data}"),
    ]

    # Create prompt parameters
    name_param = PromptParameter(
        name="name", description="Name to greet", default="World"
    )

    data_param = PromptParameter(
        name="data", description="Data to analyze", required=True
    )

    # Create prompt instances
    greet_prompt = StaticPrompt(
        name="greet",
        description="Simple greeting prompt",
        messages=[greet_message],
        arguments=[name_param],
    )

    analyze_prompt = StaticPrompt(
        name="analyze",
        description="Analysis prompt",
        messages=analyze_messages,
        arguments=[data_param],
    )

    return Config(prompts={"greet": greet_prompt, "analyze": analyze_prompt})


@pytest.mark.asyncio
async def test_prompt_command_simple(runtime_config: Config) -> None:
    """Test executing a simple prompt without arguments."""
    messages = []

    # Create output writer to capture output
    class TestOutput(DefaultOutputWriter):
        async def print(self, message: str) -> None:
            messages.append(message)

    async with RuntimeConfig.from_config(runtime_config) as runtime:
        agent: LLMlingAgent[Any] = LLMlingAgent(runtime)
        session = AgentChatSession(agent)

        # Create command context directly
        context = CommandContext(output=TestOutput(), session=session)

        # Execute prompt command
        await prompt_cmd.execute(ctx=context, args=["greet"])

    # Verify output
    assert len(messages) == 1
    assert "Hello World" in messages[0]


@pytest.mark.asyncio
async def test_prompt_command_with_args(runtime_config: Config) -> None:
    """Test executing a prompt with arguments."""
    messages = []

    class TestOutput(DefaultOutputWriter):
        async def print(self, message: str) -> None:
            messages.append(message)

    async with RuntimeConfig.from_config(runtime_config) as runtime:
        agent: LLMlingAgent[Any] = LLMlingAgent(runtime)
        session = AgentChatSession(agent)

        # Create command context directly
        context = CommandContext(output=TestOutput(), session=session)

        # Execute prompt command with arguments
        kwargs = {"data": "test.txt"}
        await prompt_cmd.execute(ctx=context, args=["analyze"], kwargs=kwargs)

    # Verify output
    assert len(messages) == 2  # noqa: PLR2004
    assert "Analyzing test.txt" in messages[0]
    assert "Please check test.txt" in messages[1]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
