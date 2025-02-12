from __future__ import annotations

from typing import Any, Literal

from pydantic_ai import RunContext
import pytest

from llmling_agent import Agent, AgentContext, AgentPool
from llmling_agent.config.capabilities import Capabilities


MODEL = "openai:gpt-4o-mini"


async def run_ctx_tool(ctx: RunContext[AgentContext], arg: str) -> str:
    """Tool expecting RunContext."""
    assert isinstance(ctx, RunContext)
    assert isinstance(ctx.deps, AgentContext)
    return f"RunContext tool got: {arg}"


async def agent_ctx_tool(ctx: AgentContext, arg: str) -> str:
    """Tool expecting AgentContext."""
    assert isinstance(ctx, AgentContext)
    return f"AgentContext tool got: {arg}"


async def data_with_run_ctx(ctx: RunContext[AgentContext]) -> str:
    """Tool accessing data through RunContext."""
    return f"Data from RunContext: {ctx.deps.data}"


async def data_with_agent_ctx(ctx: AgentContext) -> str:
    """Tool accessing data through AgentContext."""
    return f"Data from AgentContext: {ctx.data}"


async def no_ctx_tool(arg: str) -> str:
    """Tool without any context."""
    return f"No context tool got: {arg}"


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
    async with Agent[None](model=MODEL) as agent:
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
        assert deps_received.node_name == "test_agent"


@pytest.mark.asyncio
async def test_plain_tool_no_context():
    """Test that plain tools work without context."""

    async def plain_tool(arg: str) -> str:
        """Tool without context parameter."""
        return f"Got arg: {arg}"

    async with Agent[None](model=MODEL) as agent:
        agent.context = AgentContext.create_default("test_agent")
        agent.tools.register_tool(plain_tool, enabled=True)

        # Should work without error
        result = await agent.run("Use the plain_tool with arg='test'")
        assert "test" in str(result.content)


@pytest.mark.integration
@pytest.mark.parametrize("provider", ["pydantic_ai", "litellm"])
async def test_capability_tools(provider: Literal["pydantic_ai", "litellm"]):
    """Test that capability tools work with AgentContext."""
    async with AgentPool[None]() as pool:
        agent = await pool.add_agent(
            name="test_agent",
            provider=provider,
            model=MODEL,
            capabilities=Capabilities(can_list_agents=True),
        )
        result = await agent.run(
            "Get available agents using the list_agents tool and return all names."
        )
        assert agent.name in str(result.content)

        agent_2 = await pool.add_agent(
            name="test_agent_2",
            provider=provider,
            model=MODEL,
            capabilities=Capabilities(can_delegate_tasks=True),
        )

        await pool.add_agent(
            "helper",
            system_prompt="You help with tasks",
            model=MODEL,
            provider=provider,
        )
        result = await agent_2.run("Delegate 'say hello' to agent with name `helper`")
        assert "hello" in str(result.content).lower()


async def test_team_creation():
    """Test that an agent can create other agents and form them into a team."""
    async with AgentPool[None]() as pool:
        # Create orchestrator agent with needed capabilities
        caps = Capabilities(can_add_agents=True, can_add_teams=True)
        orchestrator = await pool.add_agent(
            name="orchestrator",
            model=MODEL,
            capabilities=caps,
        )

        # Ask it to create a content team
        result = await orchestrator.run("""
            Create two agents:
            1. A researcher who finds information named "alice"
            2. A writer who creates content named "bob"
            Then create a sequential team named "crew" with these agents.
        """)

        # Verify agents were created
        assert "alice" in pool.agents
        assert "bob" in pool.agents
        assert "crew" in pool.teams
        # Verify team creation message
        assert "alice" in str(result.content.lower())
        assert "bob" in str(result.content.lower())


@pytest.mark.asyncio
async def test_context_compatibility():
    """Test that both context types work in tools."""
    async with Agent[None](model=MODEL) as agent:
        agent.tools.register_tool(run_ctx_tool, name_override="run_ctx_tool")
        agent.tools.register_tool(agent_ctx_tool, name_override="agent_ctx_tool")
        agent.tools.register_tool(no_ctx_tool, name_override="no_ctx_tool")

        # All should work
        result1 = await agent.run("Use run_ctx_tool with argument 'test'")
        assert any(
            call.result == "RunContext tool got: test" for call in result1.tool_calls
        )

        result2 = await agent.run("Use agent_ctx_tool with argument 'test'")
        assert any(
            call.result == "AgentContext tool got: test" for call in result2.tool_calls
        )

        result3 = await agent.run("Use no_ctx_tool with argument 'test'")
        assert any(
            call.result == "No context tool got: test" for call in result3.tool_calls
        )


@pytest.mark.asyncio
async def test_context_sharing():
    """Test that both context types access same data."""
    shared_data = {"key": "value"}

    agent = Agent[dict](name="test", model=MODEL)
    agent.context.data = shared_data

    agent.tools.register_tool(data_with_run_ctx)
    agent.tools.register_tool(data_with_agent_ctx)

    async with agent:
        result1 = await agent.run("Use data_with_run_ctx tool")
        result2 = await agent.run("Use data_with_agent_ctx tool")

        assert any(
            call.result == "Data from RunContext: {'key': 'value'}"
            for call in result1.tool_calls
        )
        assert any(
            call.result == "Data from AgentContext: {'key': 'value'}"
            for call in result2.tool_calls
        )


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "--log-level", "debug"])
