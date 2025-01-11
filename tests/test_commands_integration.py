"""Integration tests for prompt commands."""

from __future__ import annotations

from llmling import Config, PromptMessage, PromptParameter, StaticPrompt
import pytest
from slashed import CommandStore, DefaultOutputWriter

from llmling_agent import Agent, AgentPoolView
from llmling_agent_commands.prompts import prompt_cmd


@pytest.fixture
def config() -> Config:
    """Create a Config with test prompts."""
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
async def test_prompt_command_simple(config: Config):
    """Test executing a simple prompt without arguments."""
    messages = []

    class TestOutput(DefaultOutputWriter):
        async def print(self, message: str):
            messages.append(message)

    async with Agent[None].open(config) as agent:
        session = AgentPoolView(agent)

        store = CommandStore(enable_system_commands=True)
        context = store.create_context(session, output_writer=TestOutput())

        # Execute prompt command
        await prompt_cmd.execute(ctx=context, args=["greet"])

        # Verify message was added to conversation history
        history = agent.conversation.get_history()
        assert len(history) == 1
        message = history[0]
        assert "prompt:greet" in str(message)
        assert "Hello World" in str(message)

        # Verify user feedback
        assert len(messages) == 1
        assert "Added prompt 'greet' to next message" in messages[0]


@pytest.mark.asyncio
async def test_prompt_command_with_args(config: Config):
    """Test executing a prompt with arguments."""
    messages = []

    class TestOutput(DefaultOutputWriter):
        async def print(self, message: str):
            messages.append(message)

    async with Agent[None].open(config) as agent:
        session = AgentPoolView(agent)

        store = CommandStore(enable_system_commands=True)
        context = store.create_context(session, output_writer=TestOutput())

        # Execute prompt command with arguments
        kwargs = {"data": "test.txt"}
        await prompt_cmd.execute(ctx=context, args=["analyze"], kwargs=kwargs)

        # Verify message was added to conversation history
        history = agent.conversation.get_history()
        assert len(history) == 1

        # Get content from each part
        content = str(history[0])
        assert "prompt:analyze" in content
        assert "Analyzing test.txt" in content
        assert "Please check test.txt" in content

        # Verify user feedback
        assert len(messages) == 1
        assert "Added prompt 'analyze' to next message" in messages[0]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
