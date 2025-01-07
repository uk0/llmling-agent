from __future__ import annotations

from llmling import Config
from pydantic_ai import RunContext
import pytest

from llmling_agent.agent import Agent
from llmling_agent.models.context import AgentContext


@pytest.mark.asyncio
async def test_tool_context_injection():
    """Test that tools receive correct context."""
    context_received = None
    deps_received = None

    async def test_tool(ctx: RunContext[AgentContext], arg: str) -> str:
        """Test tool that captures its context."""
        nonlocal context_received, deps_received
        context_received = ctx
        deps_received = ctx.deps
        return f"Got arg: {arg}"

    # Create minimal config
    config = Config()

    # Create agent with dependencies
    test_deps = {"key": "value"}
    context = AgentContext.create_default("test_agent")
    context.data = test_deps
    async with Agent.open(config, model="openai:gpt-4o-mini") as agent:
        agent.context = context
        # Register our test tool
        agent.tools.register_tool(test_tool, enabled=True)
        # Run agent which should trigger tool
        await agent.run("Use the test_tool with arg='test'")

        # Verify context
        assert context_received is not None, "Tool did not receive context"
        assert isinstance(context_received, RunContext), "Wrong context type"

        # Verify dependencies
        assert deps_received is not None, "Tool did not receive dependencies"
        assert deps_received.data == test_deps, "Wrong dependencies received"

        # Verify agent context
        assert deps_received.agent_name == "test_agent"


@pytest.mark.asyncio
async def test_plain_tool_no_context():
    """Test that plain tools work without context."""

    async def plain_tool(arg: str) -> str:
        """Tool without context parameter."""
        return f"Got arg: {arg}"

    config = Config()

    async with Agent.open(config, model="openai:gpt-4o-mini") as agent:
        agent.context = AgentContext.create_default("test_agent")
        agent.tools.register_tool(plain_tool, enabled=True)

        # Should work without error
        result = await agent.run("Use the plain_tool with arg='test'")
        assert "test" in str(result.content)
