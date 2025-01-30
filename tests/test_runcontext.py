from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext
import pytest

from llmling_agent.agent import Agent
from llmling_agent.config.capabilities import Capabilities
from llmling_agent.delegation.pool import AgentPool
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
    # Create agent with dependencies
    test_deps = {"key": "value"}
    context = AgentContext[Any].create_default("test_agent")
    context.data = test_deps
    async with Agent[None].open(model="openai:gpt-4o-mini") as agent:
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

    async with Agent[None](model="openai:gpt-4o-mini") as agent:
        agent.context = AgentContext.create_default("test_agent")
        agent.tools.register_tool(plain_tool, enabled=True)

        # Should work without error
        result = await agent.run("Use the plain_tool with arg='test'")
        assert "test" in str(result.content)


@pytest.mark.integration
async def test_capability_tools():
    """Test that capability tools work with AgentContext."""
    async with AgentPool[None]() as pool:
        agent = await pool.add_agent(
            name="test_agent",
            provider="pydantic_ai",
            model="gpt-4o-mini",
            capabilities=Capabilities(can_list_agents=True),
        )
        result = await agent.run(
            "Get available agents using the list_agents tool and return all names."
        )
        assert agent.name in str(result.content)

        agent_2 = await pool.add_agent(
            name="test_agent_2",
            provider="pydantic_ai",
            model="gpt-4o-mini",
            capabilities=Capabilities(can_delegate_tasks=True),
        )

        await pool.add_agent(
            "helper", system_prompt="You help with tasks", model="gpt-4o-mini"
        )
        result = await agent_2.run("Delegate 'say hello' to agent with name `helper`")
        assert "hello" in str(result.content).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "--log-level", "debug"])
